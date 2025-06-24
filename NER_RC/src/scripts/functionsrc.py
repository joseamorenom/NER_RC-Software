#DOS_Clase dinamica
# -*- coding: utf-8 -*-
"""
functions_rc.py

Created on [Fecha]
Autor: sanmo y jammm

Este módulo integra las funciones principales del back-end de NER RC:
  - Funciones utilitarias (check_create, str2bool, load_rel2id)
  - Definición del modelo (define_model) usando la red CNN
  - Preparación del dataset mediante processMatriz (para archivos JSON de formato Pratech)
  - Entrenamiento (train_model_rc) con ciclo, early stopping y guardado del modelo + rel2id
  - Uso del modelo (use_model_rc) para realizar inferencia en documentos de prueba
"""

import os
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import flair
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

#----------------------------------------------------------------
# 1. Inicialización de los parámetros globales
#----------------------------------------------------------------
default_path = None
train_loader = None
val_loader = None
test_loader = None
cnn = None
optimizer = None
criterion = None
device = None
learning_rate = 0.001
best_valid_loss = float('inf')

#----------------------------------------------------------------
# 2. Funciones utilitarias
#----------------------------------------------------------------
def load_rel2id(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"rel2id file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        rel2id = json.load(f)
    if not isinstance(rel2id, dict):
        raise ValueError("rel2id JSON must be a dict")
    id2rel = {v: k for k, v in rel2id.items()}
    return rel2id, id2rel

def check_create(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#----------------------------------------------------------------
# 3. Definición del modelo CNN
#----------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes, kernel_size=(3,1)):
        super().__init__()
        self.conv_head  = nn.Conv2d(1,1,kernel_size)
        self.conv_inter = nn.Conv2d(1,1,kernel_size)
        self.conv_tail  = nn.Conv2d(1,1,kernel_size)
        self.relu1 = nn.ReLU(); self.relu2 = nn.ReLU(); self.relu3 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d((5,1), stride=1)
        self.maxpool_2 = nn.MaxPool2d((3,1), stride=1)
        self.maxpool_3 = nn.MaxPool2d((9,1), stride=1)
        self.fc1 = nn.Linear(5*1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x_head, x_ent1, x_inter, x_ent2, x_tail):
        out_h = self.relu1(self.conv_head(x_head)); out_h = self.maxpool_1(out_h)[:,0,0,:]
        out_i = self.relu2(self.conv_inter(x_inter)); out_i = self.maxpool_2(out_i)[:,0,0,:]
        out_t = self.relu3(self.conv_tail(x_tail)); out_t = self.maxpool_3(out_t)[:,0,0,:]
        x_ent1 = torch.tanh(x_ent1); x_ent2 = torch.tanh(x_ent2)
        comb = torch.cat([out_h, x_ent1, out_i, x_ent2, out_t], dim=1)
        comb = self.fc1(comb); comb = self.fc2(comb); comb = self.fc3(comb)
        return comb

#----------------------------------------------------------------
# 4. Dataset processMatriz (sin cambios)
#----------------------------------------------------------------
class processMatriz(Dataset):
    def __init__(self, input_data):
        """
        Clase para procesar matrices a partir de datos JSON ya cargados.

        Parámetros:
          - input_data: dict
              Diccionario que contiene las claves: "flat_emb", "relation", "h_pos" y "t_pos".
              Se asume que ya está parseado (por ejemplo, mediante json.load).
        """
        input_data = input_data[0]
        if not isinstance(input_data, dict):
            raise ValueError("Se esperaba un diccionario con los datos, pero se recibió otro tipo.")

        # Guardamos el diccionario de datos
        data = input_data

        # Inicialización de los atributos que almacenarán los tensores procesados
        self.inic_m = None
        self.head_m = None
        self.inter_m = None
        self.tail_m = None
        self.final_m = None
        self.relation = None

        # Función auxiliar para agregar padding a un tensor
        def padding(tensor, tamano_deseado):
            tamano_actual = tensor.size(0)
            filas_faltantes = tamano_deseado - tamano_actual
            if filas_faltantes > 0:
                ceros_a_agregar = torch.zeros(filas_faltantes, tensor.size(1))
                tensor = torch.cat((tensor, ceros_a_agregar), dim=0)
            return tensor

        # Se extraen las claves del diccionario
        flat_emb = data["flat_emb"]
        relation = data["relation"]
        h_pos_t = data["h_pos"]
        t_pos_t = data["t_pos"]

        flat_emb_t = torch.tensor(flat_emb)
        # Se reorganiza la matriz de embeddings. Se asume que cada fila es un embedding concatenado.
        Matriz3d = flat_emb_t.reshape((flat_emb_t.shape[0], 1, -1, 1024))

        # Inicialización de tensores vacíos para cada segmento de la oración
        inic_m = torch.empty(0, 7, 1024)
        head_m = torch.empty(0, 1024)
        inter_m = torch.empty(0, 5, 1024)
        tail_m = torch.empty(0, 1024)
        final_m = torch.empty(0, 11, 1024)

        count = 0
        for matrizsub in Matriz3d:
            inic = matrizsub[0][0:h_pos_t[count][0]][:]
            if len(h_pos_t[count]) == 1:
                head = matrizsub[0][h_pos_t[count][0]][:]
                inter = matrizsub[0][h_pos_t[count][0] + 1:t_pos_t[count][0]][:]
                if len(t_pos_t[count]) == 1:
                    tail = matrizsub[0][t_pos_t[count][0]][:]
                    final = matrizsub[0][t_pos_t[count][0] + 1:][:]
                else:
                    tail = matrizsub[0][t_pos_t[count][0]:t_pos_t[count][-1] + 1][:]
                    final = matrizsub[0][t_pos_t[count][-1] + 1:][:]
            else:
                head = matrizsub[0][h_pos_t[count][0]:h_pos_t[count][-1] + 1][:]
                inter = matrizsub[0][h_pos_t[count][-1] + 1:t_pos_t[count][0]][:]
                if len(t_pos_t[count]) == 1:
                    tail = matrizsub[0][t_pos_t[count][0]][:]
                    final = matrizsub[0][t_pos_t[count][0] + 1:][:]
                else:
                    tail = matrizsub[0][t_pos_t[count][0]:t_pos_t[count][-1] + 1][:]
                    final = matrizsub[0][t_pos_t[count][-1] + 1:][:]
            count += 1

            # Si head o tail tienen más de una fila, se promedian
            if head.dim() != 1:
                head = torch.mean(head, dim=0)
            if tail.dim() != 1:
                tail = torch.mean(tail, dim=0)

            # Aplicar padding si es necesario
            if inic.size(0) > 7:
                inic = inic[:7]
            elif inic.size(0) < 7:
                inic = padding(inic, 7)
            if inter.size(0) > 5:
                inter = inter[:5]
            elif inter.size(0) < 5:
                inter = padding(inter, 5)
            if final.size(0) > 11:
                final = final[:11]
            elif final.size(0) < 11:
                final = padding(final, 11)

            # Concatenar cada muestra a los tensores finales
            inic_m = torch.cat((inic_m, inic.unsqueeze(0)), dim=0)
            head_m = torch.cat((head_m, head.unsqueeze(0)), dim=0)
            inter_m = torch.cat((inter_m, inter.unsqueeze(0)), dim=0)
            tail_m = torch.cat((tail_m, tail.unsqueeze(0)), dim=0)
            final_m = torch.cat((final_m, final.unsqueeze(0)), dim=0)

        tam = inic_m.size(0)
        self.inic_m = inic_m.reshape(tam, 1, 7, 1024)
        self.head_m = head_m.reshape(tam, 1024)
        self.inter_m = inter_m.reshape(tam, 1, 5, 1024)
        self.tail_m = tail_m.reshape(tam, 1024)
        self.final_m = final_m.reshape(tam, 1, 11, 1024)
        self.relation = np.array(relation).reshape(-1, 1)

    def __len__(self):
        """Retorna el número de muestras en el dataset."""
        return self.inic_m.shape[0]

    def __getitem__(self, index):
        x_inic  = self.inic_m[index]
        x_head  = self.head_m[index]
        x_inter = self.inter_m[index]
        x_tail  = self.tail_m[index]
        x_final = self.final_m[index]
        # Convertimos la etiqueta a LongTensor
        label = torch.tensor(int(self.relation[index]), dtype=torch.long)
        return x_inic, x_head, x_inter, x_tail, x_final, label


#----------------------------------------------------------------
# 5. Funciones de transformación Pratech → embeddings (sin cambios)
#----------------------------------------------------------------
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def insert_tags(text, b1, e1, b2, e2):
    """
    Inserta las etiquetas de forma robusta, reconstruyendo la cadena.
    Asume que las entidades no se solapan.
    """
    if b1 > b2:
        b1, e1, b2, e2 = b2, e2, b1, e1
        tag1, tag2 = ("<e2>", "</e2>"), ("<e1>", "</e1>")
    else:
        tag1, tag2 = ("<e1>", "</e1>"), ("<e2>", "</e2>")

    part1 = text[:b1]
    entity1 = text[b1:e1]
    part2 = text[e1:b2]
    entity2 = text[b2:e2]
    part3 = text[e2:]
    
    return f"{part1}{tag1[0]}{entity1}{tag1[1]}{part2}{tag2[0]}{entity2}{tag2[1]}{part3}"

'''
def limpiar_pos(frase):
    entidad1_pos = []
    entidad2_pos = []
    indice = 0
    while indice < len(frase) and frase[indice].isdigit():
        indice += 1
    frase = frase[indice:].lstrip('\t')
    frase = frase.replace(".", "").replace(",", "").replace(";", "").replace("'s","")
    if frase.startswith('"') and frase.endswith('"'):
        frase = frase[1:-1]
    if frase.endswith('.'):
        frase = frase[:-1]
    inicio_entidad1 = frase.find("<e1>")
    fin_entidad1 = frase.find("</e1>")
    if inicio_entidad1 != -1 and fin_entidad1 != -1:
        entidad1_pos = [inicio_entidad1 - 1, fin_entidad1 - len("<e1>") - 2]
        entidad1_pos = [x+1 for x in entidad1_pos]
        frase = frase.replace("<e1>", "").replace("</e1>", "")
    inicio_entidad2 = frase.find("<e2>")
    fin_entidad2 = frase.find("</e2>")
    if inicio_entidad2 != -1 and fin_entidad2 != -1:
        entidad2_pos = [inicio_entidad2 - 1, fin_entidad2 - len("<e2>") - 2]
        entidad2_pos = [x+1 for x in entidad2_pos]
        frase = frase.replace("<e2>", "").replace("</e2>", "")
    return frase, entidad1_pos, entidad2_pos
'''

def limpiar_pos(frase):
    """
    Toma una cadena con formato:
      123    <e1>EntidadA</e1> texto <e2>EntidadB</e2>...
    y devuelve:
      - clean: la frase sin tags ni puntuación (opcional)
      - entidad1_pos: [inicio, fin] en caracteres de EntidadA
      - entidad2_pos: [inicio, fin] en caracteres de EntidadB
    """
    # 1) Quitar prefijo numérico y tabulación
    idx = 0
    while idx < len(frase) and frase[idx].isdigit():
        idx += 1
    s = frase[idx:].lstrip('\t')

    # 2) Localizar posiciones en caracteres con los tags presentes
    i1_open  = s.find("<e1>")
    i1_close = s.find("</e1>") - len("<e1>")
    i2_open  = s.find("<e2>")
    i2_close = s.find("</e2>") - len("<e2>")

    # 3) Construir las listas de posiciones (si no se encuentran, quedan vacías)
    if i1_open == -1 or i1_close < i1_open:
        entidad1_pos = []
    else:
        entidad1_pos = [i1_open, i1_close]
    if i2_open == -1 or i2_close < i2_open:
        entidad2_pos = []
    else:
        entidad2_pos = [i2_open, i2_close]

    # 4) Eliminar los tags para obtener el texto limpio
    clean = (s
        .replace("<e1>", "")
        .replace("</e1>", "")
        .replace("<e2>", "")
        .replace("</e2>", "")
    )

    # 5) (Opcional) Eliminar puntuación sin alterar índices previos
    clean = clean.replace(".", "").replace(",", "").replace(";", "").replace("'s", "")

    return clean, entidad1_pos, entidad2_pos

def combinar_listas(lista):
    lista_combinada = []
    for sublist in lista:
        lista_combinada.extend(sublist)
    return lista_combinada

def sacar_embedding(frase):
    embedding_final = []
    # Se instancia TransformerWordEmbeddings cada vez; en producción podrías instanciarlo una sola vez.
    embedding = TransformerWordEmbeddings('xlm-roberta-large')
    sentence = Sentence(frase)
    embedding.embed(sentence)
    for token in sentence:
        embedding_final.append(token.embedding.tolist())
    embedding_final = combinar_listas(embedding_final)
    return embedding_final

def agregar_ceros(lista):
    max_longitud = max(len(sublista) for sublista in lista)
    for sublista in lista:
        if len(sublista) < max_longitud:
            sublista.extend([0] * (max_longitud - len(sublista)))
    return lista

def obtener_posicion_palabras(frase, posa, posb):
    sentence = Sentence(frase)
    tokens = sentence.tokens
    pos1 = int(posa[0])
    pos2 = int(posb[0])
    pos3 = int(posa[1])
    pos4 = int(posb[1])
    posicion_palabra1 = 0
    posicion_palabra2 = 0
    posicion_palabra1f = 0
    posicion_palabra2f = 0
    for i, token in enumerate(tokens):
        if token.start_position == pos1:
            posicion_palabra1 = i
        if token.end_position - 1 == pos3:
            posicion_palabra1f = i
        if token.start_position == pos2:
            posicion_palabra2 = i
        if token.end_position - 1 == pos4:
            posicion_palabra2f = i
    salida1 = [posicion_palabra1] if posicion_palabra1 == posicion_palabra1f else [posicion_palabra1, posicion_palabra1f]
    salida2 = [posicion_palabra2] if posicion_palabra2 == posicion_palabra2f else [posicion_palabra2, posicion_palabra2f]
    return salida1, salida2


def process_json_data(json_data):
    output_lines = []
    global_index = 1

    # Cargar el texto de cada oración en un diccionario
    Frases_completas = {s['id']: s['text'] for s in json_data.get('sentences', [])}
    
    # --- LA CLAVE ---
    # Cargar la posición de inicio de cada oración en un diccionario
    Posiciones_inicio_frases = {s['id']: s['begin'] for s in json_data.get('sentences', [])}
    
    # Cargar las posiciones globales de cada mención
    Mentions = {m['id']: (m['begin'], m['end']) for m in json_data.get('mentions', [])}

    if 'relations' in json_data:
        for relation in json_data['relations']:
            relation_id = relation['id']
            relation_type = relation['type']
            args = relation['args']

            arg1, arg2 = args
            
            # Identificar la oración a la que pertenecen las menciones
            sentence_id = arg1.split('-')[0]
            sentence_text = Frases_completas.get(sentence_id)
            
            # Si por alguna razón la oración no existe, saltar esta relación
            if not sentence_text:
                continue

            # Obtener la posición de inicio GLOBAL de la oración actual
            sentence_start_offset = Posiciones_inicio_frases.get(sentence_id, 0)

            # Obtener las posiciones GLOBALES de las menciones
            begin1_global, end1_global = Mentions[arg1]
            begin2_global, end2_global = Mentions[arg2]

            # Convertir las posiciones GLOBALES a LOCALES (relativas a la oración)
            begin1_local = begin1_global - sentence_start_offset
            end1_local = end1_global - sentence_start_offset
            begin2_local = begin2_global - sentence_start_offset
            end2_local = end2_global - sentence_start_offset
            
            # Insertar las etiquetas usando las posiciones locales y correctas
            tagged_sentence = insert_tags(sentence_text, begin1_local, end1_local, begin2_local, end2_local)

            output_lines.append(f"{global_index}\t\"{tagged_sentence}\"\n")
            output_lines.append(f"{relation_type}\n")
            output_lines.append(f"Comment: {relation_id}\n\n")
            global_index += 1
    else:
        print("No se encontraron 'relations' en el archivo JSON.")

    return output_lines

def process_single_file(json_file_path): #Quite output_file_path de la entrada

    if not json_file_path.endswith('.json'):
        print("El archivo proporcionado no es un JSON válido.")
        return

    if not os.path.isfile(json_file_path):
        print("El archivo no existe:", json_file_path)
        return

    print(f"Procesando archivo: {json_file_path}")

    # Leer y procesar el archivo JSON
    json_data = read_json_file(json_file_path)
    output_lines = process_json_data(json_data)

    # Escribir el archivo de salida
    #with open(output_file_path, 'w', encoding='utf-8') as file:
    #    file.writelines(output_lines)

    return output_lines

    print("Archivo de salida generado:", output_file_path)


#Para sacar las relaciones
def numero_relacion(relacion, dic):
    relacion = relacion.split("(")[0]
    return dic.get(relacion, None)

# Función única que realiza todo el proceso.
def process_document(input_data, rel2id):
    """
    Procesa un conjunto de datos (como string o lista de líneas) y retorna
    un diccionario con la información extraída (embeddings, relaciones, posiciones).

    Parámetros:
      - input_data: Puede ser un string (con saltos de línea) o una lista de líneas,
                    que representa el contenido del archivo de entrada.

    El proceso:
      1. Itera sobre las líneas del input.
      2. Para cada línea que corresponde a un token (líneas que empiezan con dígito):
             - Limpia la oración (limpiar_pos)
             - Obtiene los embeddings (sacar_embedding)
             - Extrae las posiciones de las entidades (obtener_posicion_palabras)
      3. Para las líneas que contienen la relación, se procesa con numero_relacion.
      4. Ajusta los embeddings con agregar_ceros.
      5. Arma y retorna un diccionario con las llaves: 'flat_emb', 'relation', 'h_pos' y 't_pos'.
    """
    # Si input_data es un string, se separa en líneas; si ya es lista, se usa directamente.
    if isinstance(input_data, str):
        lineas = input_data.splitlines()
    else:
        lineas = input_data

    embeddings_list = []
    relaciones = []
    posiciones1 = []
    posiciones2 = []

    # Variables temporales para procesar una oración
    fraseLimpia = ''
    relacion = 0
    pos1 = []
    pos2 = []

    for linea in lineas:
        linea = linea.strip()
        # Si la línea comienza con "Comment:", reinicia la oración
        if linea.startswith('Comment:'):
            fraseLimpia = ''
            relacion = 0
        elif not linea:
            continue  # Ignorar líneas vacías
        else:
            # Si la línea empieza con dígito, se asume que es parte de la oración
            if linea[0].isdigit():
                # Limpia la oración y obtiene las posiciones de las entidades
                fraseLimpia, pos1, pos2 = limpiar_pos(linea)
                # Obtiene el embedding de la oración limpia
                embedding1 = sacar_embedding(fraseLimpia)
                embeddings_list.append(embedding1)
                # Extrae las posiciones de los tokens para las entidades
                hpos, tpos = obtener_posicion_palabras(fraseLimpia, pos1, pos2)
                posiciones1.append(hpos)
                posiciones2.append(tpos)
            else:
                # Si la línea no comienza con dígito, se asume que es la relación
                relacion = numero_relacion(linea, rel2id)
                relaciones.append(relacion)

    # Ajustar los embeddings para que todas tengan la misma longitud
    embeddings_list = agregar_ceros(embeddings_list)

    # Armar el diccionario final
    dicc_final = {
        'flat_emb': embeddings_list,
        'relation': relaciones,
        'h_pos': posiciones1,
        't_pos': posiciones2
    }

    return dicc_final


#----------------------------------------------------------------
# 6. define_model ahora recibe num_classes
#----------------------------------------------------------------
def define_model(num_classes):
    global cnn, optimizer, criterion, device, learning_rate
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flair.device = device
    cnn = CNN(num_classes).to(device)
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

#----------------------------------------------------------------
# 7. prepare_training_data recibe rel2id para validar
#----------------------------------------------------------------
def prepare_training_data(json_file_path_t, rel2id):
    global train_loader, val_loader
    datos_sem = process_single_file(json_file_path_t)

    # validar que todas las relaciones estén en rel2id:
    for line in datos_sem:
        if not line.startswith("Comment:") and not line[0].isdigit():
            line = line.strip()          # quita espacios y saltos de línea de ambos extremos
            rel = line.split("(")[0]
            if rel not in rel2id:
                raise ValueError(f"Relación desconocida '{rel}' en datos de entrenamiento")

    datos_emb = process_document(datos_sem,rel2id)
    # Después de datos_emb = process_document(...)
    # -------------------------------------------------
    # filtramos los ejemplos cuyo 'relation' sea None
    good_idx = [i for i, r in enumerate(datos_emb['relation']) if r is not None]
    datos_emb = {
        'flat_emb': [ datos_emb['flat_emb'][i]   for i in good_idx ],
        'relation': [ datos_emb['relation'][i]   for i in good_idx ],
        'h_pos':    [ datos_emb['h_pos'][i]      for i in good_idx ],
        't_pos':    [ datos_emb['t_pos'][i]      for i in good_idx ],
    }
    
    full_dataset = processMatriz([datos_emb])
    train_size = int(0.9 * len(full_dataset)); valid_size = len(full_dataset) - train_size
    tr, va = random_split(full_dataset, [train_size, valid_size])
    train_loader = DataLoader(tr, batch_size=10, shuffle=True)
    val_loader   = DataLoader(va, batch_size=10, shuffle=True)

def prepare_test_data(json_file_path, rel2id):
    """
    Aqui toca es recibir practech, volverlo sem eval y ahi si volverlo embedding con processmatrix
    Prepara el dataset de prueba a partir de un archivo JSON usando processMatriz.
    Crea el DataLoader de test.
    """
    global test_loader
    try:
        #Primero paso de PRATECH A SEMEVAL ----------------------------------------

        # Procesar todos los archivos en la carpeta y generar el archivo de salida
        datos_sem = process_single_file(json_file_path)

        #Segundo paso de SEMEVAL A EMBEDDINGS
        #output_file_path_2 = output_file_path_t.replace("salida.json", "salida2.json")
        datos_emb = process_document(datos_sem, rel2id)
        # Después de datos_emb = process_document(...)
        # -------------------------------------------------
        # filtramos los ejemplos cuyo 'relation' sea None
        good_idx = [i for i, r in enumerate(datos_emb['relation']) if r is not None]
        datos_emb = {
            'flat_emb': [ datos_emb['flat_emb'][i]   for i in good_idx ],
            'relation': [ datos_emb['relation'][i]   for i in good_idx ],
            'h_pos':    [ datos_emb['h_pos'][i]      for i in good_idx ],
            't_pos':    [ datos_emb['t_pos'][i]      for i in good_idx ],
        }
        

        full_test_dataset = processMatriz([datos_emb])

        #Por ultimo creo el DATASET FULL
        test_loader = DataLoader(full_test_dataset, batch_size=1, shuffle=False)  #Revisar este batch
        print(f"Dataset de prueba preparado: {len(full_test_dataset)} ejemplos")
    except Exception as e:
        print("Error en prepare_test_data():", e)
        raise


#----------------------------------------------------------------
# 8. training_model_rc con copia de rel2id
#----------------------------------------------------------------
def training_model_rc(name, json_file_path, rel2id_path, epochs):
    global best_valid_loss, default_path
    best_valid_loss = float('inf')
    default_path = os.path.dirname(os.path.abspath(__file__)).replace('\\','/')
    rel2id, id2rel = load_rel2id(rel2id_path)
    num_classes = num_classes = max(rel2id.values()) + 1

    define_model(num_classes)
    check_create(f"{default_path}/../../models/RC/{name}/")
    prepare_training_data(json_file_path, rel2id)

    save_path = None
    for epoch in range(int(epochs)):
        cnn.train(); total_loss=0; total_samples=0
        for batch in train_loader:
            *X, labels = [t.to(device) for t in batch]
            # CrossEntropyLoss necesita LongTensor para los índices
            labels = labels.view(-1).long()
            optimizer.zero_grad()
            out = cnn(*X)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()*labels.size(0); total_samples += labels.size(0)
        avg_train = total_loss/total_samples

        cnn.eval(); val_loss=0; val_samples=0
        with torch.no_grad():
            for batch in val_loader:
                *X, labels = [t.to(device) for t in batch]
                labels = labels.view(-1).long()
                out = cnn(*X)
                loss = criterion(out, labels)
                val_loss += loss.item()*labels.size(0); val_samples += labels.size(0)
        avg_val = val_loss/val_samples

        if avg_val < best_valid_loss:
            best_valid_loss = avg_val
            save_path = f"{default_path}/../../models/RC/{name}/best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': cnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
            # — copia dinámicamente rel2id.json —
            shutil.copy(rel2id_path, f"{default_path}/../../models/RC/{name}/rel2id.json")
        else:
            # early stopping si quieres...
            pass

    return f"Modelo entrenado y guardado en {save_path}"

#----------------------------------------------------------------
# 9. use_model_rc recarga arquitectura según rel2id
#----------------------------------------------------------------
def use_model_rc(name, test_json_path, output_file):
    """
    Carga un modelo RC y etiqueta **todas** las frases del archivo de entrada
    dentro de sus propias N clases, IGNORANDO cualquier 'relations' que traiga el JSON.
    """
    global default_path
    default_path = os.path.dirname(os.path.abspath(__file__)).replace('\\','/')
    # 1) cargar modelo y rel2id
    model_path  = f"{default_path}/../../models/RC/{name}/best_model.pt"
    rel2id_file = f"{default_path}/../../models/RC/{name}/rel2id.json"
    rel2id, id2rel = load_rel2id(rel2id_file)
    num_classes = num_classes = max(rel2id.values()) + 1

    define_model(num_classes)
    state = torch.load(model_path, map_location=device)
    cnn.load_state_dict(state['model_state_dict'])
    cnn.eval()

    # 2) leer y unificar SemEval-lines
    try:
        datos_sem = process_single_file(test_json_path)
    except Exception:
        # no era JSON Pratech con "relations", tomamos todo como SemEval puro
        with open(test_json_path, 'r', encoding='utf-8') as f:
            datos_sem = [l.strip() for l in f if l.strip()]

    # 3) extraer frases limpias y spans de entidades
    original_sentences = []
    for line in datos_sem:
        if line and line[0].isdigit():
            _, rest = line.split('\t', 1)
            tagged = rest.strip().strip('"')
            # aquí opcionalmente recalculas e1s,e1e,e2s,e2e si lo necesitas
            original_sentences.append(tagged)

    # 4) convertir a embeddings + (ignorar relaciones del archivo)
    datos_emb = process_document(datos_sem, rel2id)
    # forzamos un dummy‐label válido para todas las instancias
    datos_emb['relation'] = [0] * len(datos_emb['flat_emb'])

    # 5) DataLoader
    dataset = processMatriz([datos_emb])
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    # 6) inferencia
    preds = []
    with torch.no_grad():
        for batch in loader:
            *X, _ = [t.to(device) for t in batch]
            out = cnn(*X)
            _, p = torch.max(out, 1)
            preds.append(p.item())

    # 7) mapear a cadenas
    mapped = [ id2rel[p] for p in preds ]

    # 8) escribir JSON final
    results = []
    for sent_tagged, rel in zip(original_sentences, mapped):
        results.append({
            "sentence": sent_tagged,
            "relation": rel
        })
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    return results

#----------------------------------------------------------------
# 10. actualizar_relaciones
#----------------------------------------------------------------
def actualizar_relaciones(numeros, id2rel):
    salida = []
    for num in numeros:
        if num not in id2rel:
            raise ValueError(f"Predicted id {num} not in rel2id")
        salida.append(id2rel[num])
    return salida

def usage_cuda_rc(cuda):
    global device
    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        flair.device = device
        if flair.device == torch.device('cpu'): return 'Error handling GPU, CPU will be used \n Processing ....'
        elif flair.device == torch.device('cuda:0'): return 'GPU detected, GPU will be used \n Processing ....'
    else:
        device = torch.device('cpu')
        flair.device = device
        return 'CPU will be  \n Processing ....'


#training_model_rc('Prueba_escritorio_guardar_usa', r"C:\Users\amesa\OneDrive\GITA\2025\NER_RC\data\RC\prat2.json", r"C:\Users\amesa\OneDrive\GITA\2025\NER_RC\data\RC\rel2id_prat.json", 1)

#use_model_rc('Prueba_escritorio_guardar_usa', r"C:\Users\amesa\OneDrive\GITA\2025\NER_RC\data\RC\prat.json", r"C:\Users\amesa\OneDrive\GITA\2025\NER_RC\data\RC\salida_Prueba_escritorio_Prat.json")