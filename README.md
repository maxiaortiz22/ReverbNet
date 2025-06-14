# ReverbNet
Red neuronal creada para estimar parámetros acústicos de una sala a partir de un audio de voz grabado en el lugar.


Exportar el environment a un archivo:

```bash
conda env export --name ReverbNet > ReverbNet.yml
```

Para crear el mismo environment desde otra compu:

```bash
conda env create -f ReverbNet.yml
```

Para activar el environment:

```bash
conda activate ReverbNet
```

Para instalar las dependencias:

```bash
pip install requirements.txt
```

Para desactivar el environment:

```bash
conda deactivate
```