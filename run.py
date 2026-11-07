import sys; sys.path.append('code')
import os
import argparse
import pandas as pd
from code import import_configs_objs
from code import DataBaseGenerator
from code import read_dataset
from code import model, reshape_data, normalize_descriptors, prediction, descriptors_err, save_exp_data, create_early_stopping_callback
from code.splits import validation_mask
import concurrent.futures
import gc
import json
from collections import Counter
from numpy import argmin
from code.data_reader import GENERATION_SUMMARY
from warnings import filterwarnings
filterwarnings("ignore")  # Suppress non-critical warnings for cleaner console output.


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    dict
        Dictionary containing parsed command-line arguments. At minimum,
        the dictionary will include the key ``"config"`` with the path
        to the experiment configuration file supplied via ``--config``.
    """
    # Initialize argument parser.
    parser = argparse.ArgumentParser()

    # Command-line arguments.
    parser.add_argument(
        "--config", help="Config file with the experiment configurations"
    )

    # Convert parsed args to a dictionary and return.
    command_line_args = vars(parser.parse_args())
    return command_line_args


def main(**kwargs):
    """
    Main entry point for end‑to‑end dataset generation and model training.

    This function performs two high‑level tasks:

    1. **Dataset generation / caching** using multiprocessing. Raw audio/RIR
       combinations are processed and written to chunked pickle files to limit
       peak RAM usage.
    2. **Per‑band training loop** for blind acoustic descriptor estimation.
       For each frequency band listed in the configuration, the function loads
       a sample of the cached dataset, prepares inputs/targets, normalizes
       descriptor targets, builds and trains the model, evaluates errors, and
       saves experiment artifacts.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments. Expected to include:
        config : str
            Path to a Python configuration file whose namespace defines all
            required experiment settings (see keys accessed in this function).

    Notes
    -----
    The configuration object is loaded dynamically via :func:`import_configs_objs`
    and must define (at least) the keys accessed below (e.g., ``files_speech_train``,
    ``files_rirs``, ``bands``, ``filters``, etc.). No validation is performed here;
    missing keys will raise a runtime exception.
    """
    # --- Load experiment configuration ---
    config_path = kwargs.pop("config")
    print(f"Loading configuration from: {config_path}")
    config = import_configs_objs(config_path)

    # --- Instantiate database generator ---
    # Ensure parameter names match the expected keys in ``config``.
    database = DataBaseGenerator(
        config['files_speech_train'], config['files_speech_test'], config['files_rirs'],
        config['to_augmentate'], config['rirs_for_training'],
        config['rirs_for_testing'], config['bands'], config['filter_type'], config['fs'],
        config['max_ruido_dB'], config['order'], config['add_noise'], config['snr'],
        config['tr_aug'], config['drr_aug'],
        # Noise options (DATA-09). Defaults reproduce the legacy setup: always pink noise,
        # no noise-free fraction. The project-wide seed makes the noise reproducible.
        noise_beta=config.get('noise_beta', [1.0, 1.0]),
        clean_fraction=config.get('clean_fraction', 0.0),
        seed=config['seed'],
    )

    # ------------------------------------------------------------------
    # Dataset generation (multiprocessing with chunked on-disk caching)
    # ------------------------------------------------------------------

    # Directory that will contain the numbered pickle chunks.
    db_dir = database.cache_path / database.db_name      # e.g., cache/BD-TR08_SNR20
    chunk_size = 50_000                                  # records per pickle

    # The summary file is written last, so it marks a complete database. A folder without
    # it was interrupted mid-generation and must not be used (it used to be taken as done).
    summary_path = db_dir / GENERATION_SUMMARY

    if summary_path.exists():
        print(f"Database already exists at: {db_dir}")
    elif db_dir.exists() and any(db_dir.iterdir()):
        raise RuntimeError(
            f"{db_dir} exists but has no {GENERATION_SUMMARY}: a previous generation was "
            f"interrupted. Delete the folder and run again."
        )
    else:
        print("Starting database generation with multiprocessing...")
        db_dir.mkdir(parents=True, exist_ok=True)        # create directory if needed

        chunk_records: list[dict] = []
        part_idx: int = 0
        n_records: int = 0
        discards: Counter = Counter()
        n_rirs = len(config['files_rirs'])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # process_single_rir() returns (records, discard counts) for one RIR.
            results_iter = executor.map(
                database.process_single_rir,
                config['files_rirs']
            )

            for n_done, (record_list, rir_discards) in enumerate(results_iter, start=1):
                # Accumulate records and discard counts from this RIR.
                chunk_records.extend(record_list)
                n_records += len(record_list)
                discards.update(rir_discards)
                if n_done % 50 == 0 or n_done == n_rirs:
                    print(f"  - {n_done}/{n_rirs} RIRs processed, {n_records} records")

                # While we have at least chunk_size records → write and clear buffer slice.
                while len(chunk_records) >= chunk_size:
                    df_part = pd.DataFrame(chunk_records[:chunk_size])
                    df_part.to_pickle(db_dir / f"{part_idx}.pkl")
                    print(f"  - Saved {part_idx}.pkl with {len(df_part)} rows")
                    part_idx += 1
                    chunk_records = chunk_records[chunk_size:]  # drop written records

        # After the pool finishes → flush any remaining records.
        if chunk_records:
            df_part = pd.DataFrame(chunk_records)
            df_part.to_pickle(db_dir / f"{part_idx}.pkl")
            print(f"  - Saved {part_idx}.pkl with {len(df_part)} remaining rows")
            part_idx += 1

        # Discards by reason (BUG-05): units are band records or whole RIR variants.
        print("Discarded items by reason:")
        for reason, count in sorted(discards.items()):
            print(f"  - {reason}: {count}")
        summary_path.write_text(json.dumps({
            'db_name': database.db_name,
            'n_rirs': n_rirs,
            'n_records': n_records,
            'n_partitions': part_idx,
            'discards': dict(sorted(discards.items())),
        }, indent=2), encoding='utf-8')

        if n_records == 0:
            print("Warning: no records were generated; database is empty.")
        else:
            print(f"Database generated in {db_dir} with {part_idx} files ({n_records} records).")

    # --- Memory cleanup of large generator object ---
    db_name = database.db_name  # retain db_name for subsequent loads
    del database
    gc.collect()

    # ------------------------------------------------------------------
    # Per-band training loop
    # ------------------------------------------------------------------
    for band in config['bands']:
        print(f"\nStarting training for band {band} Hz:")

        # Read the requested sample fraction of the dataset for this band.
        db_train = read_dataset(band, db_name, config['sample_frac'], config['random_state'], type_data='train')
        db_test = read_dataset(band, db_name, config['sample_frac'], config['random_state'], type_data='test')

        # Validation by source RIR (BUG-04): whole RIRs, with all their speech files and
        # augmentations, go to validation. Keras' validation_split took the last records,
        # so the same RIR was on both sides and early stopping was optimistic.
        val_mask = validation_mask(db_train.ReverbedAudio, config['validation_split'], config['seed'])
        db_val = db_train[val_mask]
        db_train = db_train[~val_mask]
        print(f"Records: train {len(db_train)}, validation {len(db_val)}, test {len(db_test)}")

        # Reshape for model input.
        X_train, y_train = reshape_data(list(db_train.tae.to_numpy()), list(db_train.descriptors.to_numpy()))
        X_val, y_val = reshape_data(list(db_val.tae.to_numpy()), list(db_val.descriptors.to_numpy()))
        X_test, y_test = reshape_data(list(db_test.tae.to_numpy()), list(db_test.descriptors.to_numpy()))

        # Normalize targets with the 95th percentile of the training split only (BUG-04).
        (y_train, y_val, y_test,
         T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95) = normalize_descriptors(y_train, y_val, y_test)

        # Instantiate the model.
        blind_estimation_model = model(
            config['filters'], config['kernel_size'], config['activation'],
            config['pool_size'], config['learning_rate']
        )
        # blind_estimation_model.summary()

        # Create early stopping callback.
        early_stopping_callback = create_early_stopping_callback(patience=100, monitor='val_loss', restore_best_weights=True)

        # Train the model.
        history = blind_estimation_model.fit(
            x=X_train, y=y_train,
            validation_data=(X_val, y_val),
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            callbacks=[early_stopping_callback]
        )

        # Report training results.
        best_epoch = argmin(history.history['val_loss']) + 1
        print(f"Best epoch: {best_epoch} with val_loss: {min(history.history['val_loss']):.6f}")
        print(f"Training finished at epoch: {len(history.history['loss'])}")

        # Run model predictions.
        predict = prediction(blind_estimation_model, X_test, y_test)

        # Compute descriptor errors.
        err_t30, err_c50, err_c80, err_d50 = descriptors_err(predict, y_test)

        # Save experiment artifacts.
        save_exp_data(
            config['exp_num'], band, blind_estimation_model, history, predict,
            err_t30, err_c50, err_c80, err_d50,
            T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
            X_test, y_test
        )

        # Delete large variables to free memory before next band iteration.
        del db_train, db_val, db_test, X_train, X_val, X_test, y_train, y_val, y_test
        del T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95
        del blind_estimation_model, history, predict, err_t30, err_c50, err_c80, err_d50
        gc.collect()


if __name__ == "__main__":

    # 1) Parse command-line arguments.
    kwargs = parse_args()

    # 2) Run main entry point.
    main(**kwargs)