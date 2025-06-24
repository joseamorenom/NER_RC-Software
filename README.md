# NER_RC-Software
Sistema Completo de Reconocimiento de Entidades y Clasificación de Relaciones con Gradio

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Gradio-4.x-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

### Tabla de Contenidos
1.  [Visión General](#visión-general)
2.  [Demostración Online](#demostración-online)
3.  [Instalación y Configuración](#instalación-y-configuración)
4.  [Uso](#uso)
5.  [Agradecimientos](#agradecimientos)
6.  [Licencia](#licencia)

---

### Visión General

Este proyecto es una implementación completa de un sistema para **Reconocimiento de Entidades Nombradas (NER)** y **Clasificación de Relaciones (RC)**. Utiliza modelos basados en Transformers como `xlm-roberta-large` a través del framework **Flair** para NER, y una Red Neuronal Convolucional (CNN) personalizada en **PyTorch** para RC.

Toda la funcionalidad es accesible a través de una aplicación de escritorio interactiva creada con **Gradio**.

#### Características Principales
* **Reconocimiento de Entidades (NER):**
    * Entrenamiento de modelos NER personalizados.
    * Etiquetado de frases y documentos completos en formato JSON.
* **Clasificación de Relaciones (RC):**
    * Entrenamiento de un modelo CNN para clasificar relaciones entre entidades.
    * Inferencia sobre documentos para predecir las relaciones existentes.
* **Interfaz Gráfica:** Aplicación interactiva que permite usar todas las funcionalidades sin necesidad de código.

### Demostración Online

Para una demostración rápida sin necesidad de instalación local, puedes visitar el Space original desplegado en Hugging Face:

➡️ **[Demo en Hugging Face Spaces](https://huggingface.co/spaces/SantiagoMoreno-UdeA/NER_RC)**

### Instalación y Configuración

Sigue estos pasos para configurar el proyecto en tu máquina local.

#### Paso 1: Clonar el Repositorio

Abre una terminal y clona este repositorio de GitHub.

```bash
git clone https://github.com/joseamorenom/NER_RC-Software.git
cd NER_RC-Software
```

#### Paso 2: Descargar los Modelos Pre-entrenados

> **⚠️ Paso Crítico: Descarga de Modelos**
>
> Este repositorio **no incluye** los archivos de los modelos pre-entrenados debido a su gran tamaño. Es **imprescindible** descargarlos por separado.

1.  Descarga la carpeta `models` desde el siguiente enlace de OneDrive:
    * **[Enlace de Descarga de Modelos](https://1drv.ms/f/c/ddaedc6765eff91f/EmW-9yvi2GhHrm28MRM_09MBkjBfyGLXl9Trr02k8McRxA?e=ILlFzB)**

2.  Una vez descargada, descomprime el archivo si es necesario.
3.  Mueve la carpeta `models` que acabas de descargar y **colócala dentro de la carpeta `NER_RC`** que se encuentra en el repositorio que clonaste.

La estructura final de tu carpeta debe verse así:
NER_RC-Software/
└── NER_RC/
├── models/      &lt;-- AQUÍ DEBE ESTAR LA CARPETA QUE DESCARGASTE
├── data/
├── src/
└── requirements.txt

#### Paso 3: Crear un Entorno Virtual (Recomendado)
Es una buena práctica aislar las dependencias del proyecto. Si usas `conda`, puedes hacerlo de la siguiente manera:

```bash
# 1. Crear un nuevo entorno (se recomienda Python 3.9 o 3.10)
conda create -n ner_rc_env python=3.9

# 2. Activar el entorno
conda activate ner_rc_env
```

#### Paso 4: Instalar Dependencias
Asegúrate de estar en la carpeta NER_RC (la que contiene el archivo requirements.txt) y luego instala todas las librerías necesarias.

```bash
# Navega a la carpeta correcta si no lo has hecho
cd NER_RC

# Instala los requerimientos
pip install -r requirements.txt
```
### Uso

Una vez que hayas completado la instalación y configuración, puedes ejecutar la aplicación.

1.  Desde la carpeta `NER_RC`, navega hasta la ubicación del script de la interfaz gráfica:
    ```bash
    cd src/grap
    ```

2.  Ejecuta la aplicación con Python:
    ```bash
    python app.py
    ```

3.  Se generará una URL local (ej. `http://127.0.0.1:7860`). Ábrela en tu navegador para interactuar con la interfaz de Gradio.

### Agradecimientos

Este trabajo fue financiado por Pratech Group SAS y el Comité para el Desarrollo de la Investigación (CODI) de la Universidad de Antioquia a través de las subvenciones # IAPFI23-1-01 y # PI2019-24110.

### Licencia

Este proyecto está distribuido bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.
