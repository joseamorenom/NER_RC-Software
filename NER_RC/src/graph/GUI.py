import os
import shutil
import gradio as gr
import sys
import json
import time

# Ajuste del directorio de trabajo
default_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(default_path)
sys.path.insert(0, default_path + '/../scripts')

from src.scripts.functionsner import (
    use_model, tag_sentence, json_to_txt, training_model,
    characterize_data, upsampling_data, usage_cuda, copy_data
)
from src.scripts.functionsrc import use_model_rc, training_model_rc, usage_cuda_rc

# Funciones para listar modelos dinámicamente
def models_NER():
    path = os.path.normpath(os.path.join(default_path, '../../models/NER'))
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def models_RC():
    path = os.path.normpath(os.path.join(default_path, '../../models/RC'))
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# Helper para borrar modelos cross-platform
def delete_model(model_name, model_type='NER'):
    if not model_name:
        print("No se seleccionó ningún modelo para borrar.")
        choices = models_NER() if model_type == 'NER' else models_RC()
        return gr.update(choices=choices)

    # --- PASO CRUCIAL: LIBERAR LOS MODELOS DE LA MEMORIA ---
    # Reseteamos las variables globales para que Python suelte los archivos del modelo.
    global tagger_sentence, tagger_document
    tagger_sentence = 0
    tagger_document = 0
    # Si tienes un caché similar para RC, también deberías resetearlo aquí.
    # global tagger_rc
    # tagger_rc = 0 
    print("Caché de modelos globales reseteado para liberar bloqueos.")
    # ---------------------------------------------------------

    folder = os.path.normpath(
        os.path.join(default_path, '..', '..', 'models', model_type, model_name)
    )
    
    print(f"Intentando borrar la carpeta: {folder}")

    if os.path.isdir(folder):
        # Mantenemos el bucle de reintento por si hay otros bloqueos (ej. antivirus)
        attempts = 5
        while attempts > 0:
            try:
                shutil.rmtree(folder)
                print(f"Carpeta {folder} borrada exitosamente.")
                break 
            except OSError as e:
                print(f"Error borrando {folder}: {e}. Reintentando en 0.2 segundos...")
                attempts -= 1
                time.sleep(0.2)
        
        if os.path.exists(folder):
            print(f"ERROR: No se pudo borrar la carpeta {folder} después de varios intentos.")

    elif os.path.isfile(folder):
        # Esta lógica es para archivos sueltos, la dejamos por si acaso.
        try:
            os.remove(folder)
            print(f"Archivo {folder} borrado exitosamente.")
        except OSError as e:
            print(f"Error borrando el archivo {folder}: {e}")

    # Refrescar la lista de modelos en la GUI
    print("Refrescando la lista de modelos en la interfaz.")
    choices = models_NER() if model_type == 'NER' else models_RC()
    return gr.update(choices=choices, value=None)

# -------------------- Funciones de backend --------------------

def Trainer(fast, model_name, standard, input_dir, upsampling, cuda):
    epochs = 1 if fast else 20
    cuda_info = usage_cuda(cuda)

    if standard:
        copy_data(input_dir)
    else:
        err = json_to_txt(input_dir)
        if isinstance(err, int):
            yield f'Error procesando documentos: código {err}'
    if upsampling:
        yield f"{cuda_info}\n{'-'*10} Upsampling {'-'*10}"
        entities = list(characterize_data().keys())
        to_up = [e for e, v in characterize_data().items() if v < 200]
        upsampling_data(to_up, 0.8, entities)
        yield f"{'-'*10} Training {'-'*10}"
    else:
        yield f"{cuda_info}\n{'-'*10} Training {'-'*10}"
    err = training_model(model_name, epochs)
    if isinstance(err, int):
        yield f'Error entrenando: código {err}'
    else:
        yield f'Modelo {model_name} guardado en models/{model_name}'


def Tagger_sentence(model_name, sentence, cuda):
    # Normalizar nombre
    name = os.path.basename(model_name).strip()
    cuda_info = usage_cuda(cuda)
    yield f"{cuda_info}\n{'-'*10} Tagging {'-'*10}"
    res = tag_sentence(sentence, name)
    if isinstance(res, int):
        yield f"Error {res}, revisar documentación"
    else:
        yield res['Highligth']


def Tagger_json(model_name, input_file, output_path, cuda):
    # Normalizar nombre
    name = os.path.basename(model_name).strip()
    cuda_info = usage_cuda(cuda)
    # inicializa archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'error':'error'}, f)
    yield cuda_info, {}, output_path

    res = use_model(name, input_file.name, output_path)
    if isinstance(res, int):
        yield f"Error {res}", {}, output_path
    else:
        yield {'text': res['text'], 'entities': res['entities']}, res, output_path


def Trainer_RC(fast, model_name, data_file, rel2id_file, cuda):
    epochs = 1 if fast else 200
    cuda_info = usage_cuda_rc(cuda)
    yield f"{cuda_info}\n{'-'*10} Training RC {'-'*10}"
    err = training_model_rc(model_name, data_file.name, rel2id_file.name, epochs)
    if isinstance(err, int):
        yield f'Error RC: código {err}'
    else:
        yield f'Modelo RC {model_name} guardado en models/RC/{model_name}'


def Tagger_document_RC(model_name, input_file, output_path, cuda):
    # Normalizar nombre
    name = os.path.basename(model_name).strip()
    cuda_info = usage_cuda_rc(cuda)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'error':'error'}, f)
    yield {'cuda': cuda_info}, output_path
    res = use_model_rc(name, input_file.name, output_path)
    if isinstance(res, int):
        yield {}, output_path
    else:
        yield res, output_path

# -------------------- Construcción de la GUI --------------------

def execute_GUI():
    with gr.Blocks(
        title='NER & RC System',
        css="""
            #title {font-size: 2.5em; font-weight:bold; text-align:center; margin-bottom:0;} 
            #subtitle {font-size:1.25em; text-align:center; margin-top:0; color:#555;}  
            #sub2 {font-size:1em; text-align:center; margin-top:0; color:#777;}  
        """
    ) as demo:
        gr.HTML("<div id='title'>NER y RC por <b>GITA</b> y <b>Pratec Group S.A.S.</b></div>")
        gr.HTML("<div id='subtitle'>Diseñado por José Alejandro Moreno, Santiago Moreno, Cristian Rios, Daniel Escobar y Rafael Orozco</div>")
        gr.HTML("<div id='sub2'>Named Entity Recognition(NER) and Relation Classification (RC) System.</div>")

        # -------- Pestaña NER --------
        with gr.Tab('NER'):
            with gr.Tab('Tagger'):
                # Sentence Tagger
                with gr.Tab('Sentence'):
                    with gr.Row():
                        with gr.Column(scale=1):
                            sel_model = gr.Radio(models_NER(), label='Model')
                            br = gr.Button('Refresh Models')
                            bd = gr.Button('Delete Model')
                            br.click(lambda: gr.update(choices=models_NER()), None, sel_model)
                            bd.click(lambda m: delete_model(m, 'NER'), sel_model, sel_model)
                            inp_sentence = gr.Textbox(placeholder='Enter sentence here...', label='Sentence')
                            cuda_flag = gr.Checkbox(label='CUDA', value=False)
                            btn_tag = gr.Button('Tag')
                        out_hl = gr.HighlightedText()
                    gr.Examples(
                        examples=[
                            ['CCC', "Camara de comercio de medellín. El ciudadano JAIME JARAMILLO VELEZ identificado con C.C. 12546987 ingresó al plantel el día 1/01/2022"],
                            ['CCC', "Razón Social GASEOSAS GLACIAR S.A.S, ACTIVIDAD PRINCIPAL fabricación y distribución de bebidas endulzadas"]
                        ], inputs=[sel_model, inp_sentence, cuda_flag]
                    )
                    btn_tag.click(
                        Tagger_sentence,
                        inputs=[sel_model, inp_sentence, cuda_flag],
                        outputs=out_hl
                    )

                # Document Tagger
                
                with gr.Tab('Document'):
                    with gr.Row():
                        with gr.Column(scale=1):
                            sel_model2 = gr.Radio(models_NER(), label='Model')
                            br2 = gr.Button('Refresh Models')
                            bd2 = gr.Button('Delete Model')
                            br2.click(lambda: gr.update(choices=models_NER()), None, sel_model2)
                            bd2.click(lambda m: delete_model(m, 'NER'), sel_model2, sel_model2)
                            inp_file = gr.File(label='Input data file')
                            out_path = gr.Textbox(placeholder='Enter path here...', label='Output data file path')
                            cuda2 = gr.Checkbox(label='CUDA', value=False)
                            btn_tag2 = gr.Button('Tag')
                        out_hl2 = gr.HighlightedText()
                        out_json = gr.JSON()
                        out_file = gr.File()
                    btn_tag2.click(
                        Tagger_json,
                        inputs=[sel_model2, inp_file, out_path, cuda2],
                        outputs=[out_hl2, out_json, out_file]
                    )
                '''
                with gr.Tab('Document'):
                    with gr.Row():
                        with gr.Column(scale=1):
                            sel_model2   = gr.Radio(models_NER, label='Model')
                            br2          = gr.Button('Refresh Models')
                            bd2          = gr.Button('Delete Model')
                            br2.click(lambda: gr.update(choices=os.listdir(os.path.join(default_path, 'models','NER'))),
                                    None, sel_model2)
                            bd2.click(lambda m: (
                                    os.system(f'rmdir /S /Q "{os.path.join(default_path, "models","NER", m)}"'),
                                    gr.update(choices=os.listdir(os.path.join(default_path, 'models','NER'))))[1],
                                    sel_model2, sel_model2)
                            inp_file     = gr.File(label='Input data file')
                            out_path     = gr.Textbox(placeholder='C:\\ruta\\a\\salida.json', label='Output data file path')
                            cuda2        = gr.Radio([True, False], label='CUDA', value=False)
                            btn_tag2     = gr.Button('Tag')
                        out_hl2  = gr.HighlightedText()
                        out_json = gr.JSON()
                        out_file = gr.File()
                    btn_tag2.click(
                        Tagger_json,
                        inputs=[sel_model2, inp_file, out_path, cuda2],
                        outputs=[out_hl2, out_json, out_file]
                    )
            '''
            # Trainer NER
            with gr.Tab('Trainer'):
                with gr.Row():
                    with gr.Column():
                        in_fast  = gr.Checkbox(label='Fast training', value=False)
                        in_name  = gr.Textbox(label='New model name')
                        in_std   = gr.Checkbox(label='Standard input', value=False)
                        in_dir   = gr.Textbox(label='Input data directory path')
                        in_up    = gr.Checkbox(label='Upsampling', value=False)
                        in_cuda  = gr.Checkbox(label='CUDA', value=False)
                        btn_train = gr.Button('Train')
                    out_train = gr.TextArea(label='Output')
                btn_train.click(
                    Trainer,
                    inputs=[in_fast, in_name, in_std, in_dir, in_up, in_cuda],
                    outputs=out_train
                )

        # -------- Pestaña RC --------
        with gr.Tab('RC'):
            with gr.Tab('Tagger Document'):
                with gr.Row():
                    with gr.Column(scale=1):
                        sel_rc = gr.Radio(models_RC(), label='Model')
                        brc = gr.Button('Refresh Models')
                        bdc = gr.Button('Delete Model')
                        brc.click(lambda: gr.update(choices=models_RC()), None, sel_rc)
                        bdc.click(lambda m: delete_model(m, 'RC'), sel_rc, sel_rc)
                        inp_file_rc = gr.File(label='Input data file')
                        out_path_rc = gr.Textbox(placeholder='Enter path here...', label='Output data file path (.JSON)')
                        cuda_rc = gr.Checkbox(label='CUDA', value=False)
                        btn_tag_rc = gr.Button('Tag')
                    out_json_rc = gr.JSON()
                    out_file_rc = gr.File()
                btn_tag_rc.click(
                    Tagger_document_RC,
                    inputs=[sel_rc, inp_file_rc, out_path_rc, cuda_rc],
                    outputs=[out_json_rc, out_file_rc]
                )

            # Trainer RC
            with gr.Tab('Trainer'):
                with gr.Row():
                    with gr.Column():
                        rc_fast = gr.Checkbox(label='Fast training', value=True)
                        rc_name = gr.Textbox(label='New RC model name')
                        rc_data = gr.File(label='Input train file (.TXT)')
                        rc_rel  = gr.File(label='Input rel2id file (.JSON)')
                        rc_cuda = gr.Checkbox(label='CUDA', value=False)
                        btn_train_rc = gr.Button('Train')
                    out_train_rc = gr.TextArea(label='Output RC')
                btn_train_rc.click(
                    Trainer_RC,
                    inputs=[rc_fast, rc_name, rc_data, rc_rel, rc_cuda],
                    outputs=out_train_rc
                )

        demo.queue()
        demo.launch()

if __name__ == '__main__':
    execute_GUI()
