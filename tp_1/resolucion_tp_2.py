import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def encontrar_final_linea(indexes):
    '''Reconocer cual es el pixel final de una linea que tiene más de un píxel de ancho'''
    donde_final = list(np.diff(indexes) != 1)
    donde_final.append(True)
    return indexes[donde_final]

def encontrar_inicio_linea(indexes):
    '''Reconocer cual es el pixel inicial de una linea que tiene más de un píxel de ancho'''
    donde_inicio = [True] + list(np.diff(indexes) != 1)
    return indexes[donde_inicio]

def encontrar_rango_celda(img, axis, umbral):
    '''Encontrar el rango horizontal o vertical que ocupa la celda'''
    img_ones = img.any(axis = axis)
    index_ones = np.argwhere(img_ones)
    index_ones = index_ones.reshape(len(index_ones))
    if axis == 1:
        suma_lineas = img[index_ones, :].sum(axis = axis)
    else:
        suma_lineas = img[:, index_ones].sum(axis = axis)
    index = np.argwhere(suma_lineas >= umbral)
    index = index.reshape(len(index))
    index_general = index_ones[index]

    inicio_linea = encontrar_inicio_linea(index_general)
    final_linea = encontrar_final_linea(index_general)
    rango_celda = tuple(zip(final_linea[:-1],inicio_linea[1:]))
    return rango_celda

# obtener caracteres
def get_components(subimage):
    '''Obtener lo componentes conectados de una imágen y actualizarlos para eliminar el fondo y los componentes muy pequeños'''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(subimage, 8, cv2.CV_32S)
    ix_area = stats[:,-1]> 10
    stats = stats[ix_area,:]
    num_labels_updated = len(stats) - 1
    centroids_updated = centroids[ix_area,:]
    centroids_updated = centroids_updated[1:] #eliminar fondo
    return num_labels_updated, centroids_updated

def encontrar_rango_espacio(img, axis, umbral):
    '''Encontrar el rango que ocupa un espacios en blanco'''
    img_ones = img.any(axis = axis)
    index_ones = np.argwhere(img_ones)
    index_ones = index_ones.reshape(len(index_ones))
    if axis == 1:
        suma_lineas = img[index_ones, :].sum(axis = axis)
    else:
        suma_lineas = img[:, index_ones].sum(axis = axis)
    index = np.argwhere(suma_lineas >= umbral)
    index = index.reshape(len(index))
    index_general = index_ones[index]

    inicio_linea = encontrar_inicio_linea(index_general)
    final_linea = encontrar_final_linea(index_general)
    espacios = final_linea - inicio_linea
    return espacios

#obtener espacios
def calculate_spaces(centroids, celda, umbral):
    '''Encontrar el rango entre palabras'''
    centroids = np.sort(centroids)
    lim_inicio = centroids[0]
    lim_fin = centroids[-1]
    celda_acotada = celda[:,int(np.floor(lim_inicio)): int(np.ceil(lim_fin))]
    espacios = encontrar_rango_espacio(celda_acotada == 0, 0, 37)
    return ((espacios > umbral)*1).sum()

def validacion_general(img_path):
    '''Valudación general del formulario'''
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    umbral_general = 200
    img_bw = img < umbral_general

    #Encontrar rangos celdas
    rango_celda_x = encontrar_rango_celda(img_bw, 1, 850)
    rango_celda_y_largo = encontrar_rango_celda(img_bw, 0, 400)
    rango_celda_y_corto = encontrar_rango_celda(img_bw, 0, 175)

    celda_nombre = ((img_bw[rango_celda_x[1][0]+1:rango_celda_x[1][1], rango_celda_y_largo[1][0]+1:rango_celda_y_largo[1][1]])*1*255).astype('uint8')
    celda_edad = ((img_bw[rango_celda_x[2][0]+1:rango_celda_x[2][1], rango_celda_y_largo[1][0]+1:rango_celda_y_largo[1][1]])*1*255).astype('uint8')
    celda_mail = ((img_bw[rango_celda_x[3][0]+1:rango_celda_x[3][1], rango_celda_y_largo[1][0]+1:rango_celda_y_largo[1][1]])*1*255).astype('uint8')
    celda_legajo = ((img_bw[rango_celda_x[4][0]+1:rango_celda_x[4][1], rango_celda_y_largo[1][0]+1:rango_celda_y_largo[1][1]])*1*255).astype('uint8')
    celda_comentario = ((img_bw[rango_celda_x[9][0]+1:rango_celda_x[9][1], rango_celda_y_largo[1][0]+1:rango_celda_y_largo[1][1]])*1*255).astype('uint8')
    celda_p1_si = ((img_bw[rango_celda_x[6][0]+1:rango_celda_x[6][1], rango_celda_y_corto[1][0]+1:rango_celda_y_corto[1][1]])*1*255).astype('uint8')
    celda_p2_si = ((img_bw[rango_celda_x[7][0]+1:rango_celda_x[7][1], rango_celda_y_corto[1][0]+1:rango_celda_y_corto[1][1]])*1*255).astype('uint8')
    celda_p3_si = ((img_bw[rango_celda_x[8][0]+1:rango_celda_x[8][1], rango_celda_y_corto[1][0]+1:rango_celda_y_corto[1][1]])*1*255).astype('uint8')
    celda_p1_no = ((img_bw[rango_celda_x[6][0]+1:rango_celda_x[6][1], rango_celda_y_corto[2][0]+1:rango_celda_y_corto[2][1]])*1*255).astype('uint8')
    celda_p2_no = ((img_bw[rango_celda_x[7][0]+1:rango_celda_x[7][1], rango_celda_y_corto[2][0]+1:rango_celda_y_corto[2][1]])*1*255).astype('uint8')
    celda_p3_no = ((img_bw[rango_celda_x[8][0]+1:rango_celda_x[8][1], rango_celda_y_corto[2][0]+1:rango_celda_y_corto[2][1]])*1*255).astype('uint8')

    # validar nombre
    len_char, centroids = get_components(celda_nombre)
    if len_char >= 1:
        number_spaces = calculate_spaces(centroids[:, 0], celda_nombre, 7)
    if (len_char>=1) and (len_char <= 25) and (number_spaces >= 1 ):
        print ('Nombre: OK')
    else:
        print('Nombre: Mal')

    #validar edad
    len_char, centroids = get_components(celda_edad)
    if (len_char >=2) and (len_char <=3):
        print ('Edad: OK')
    else:
        print('Edad: Mal')

    #validar mail
    len_char, centroids = get_components(celda_mail)
    if len_char >= 1:
        number_spaces = calculate_spaces(centroids[:, 0], celda_mail, 8)
    if (len_char>=1) and (len_char <= 25) and (number_spaces == 0):
        print ('Mail: OK')
    else:
        print('Mail: Mal')

    #validar legajo
    len_char, centroids = get_components(celda_legajo)
    if len_char >= 1:
        number_spaces = calculate_spaces(centroids[:, 0], celda_legajo, 8)
    if (len_char == 8) and (number_spaces == 0):
        print ('Legajo: OK')
    else:
        print('Legajo: Mal')

    #validar comentarios
    len_char, centroids = get_components(celda_comentario)
    if len_char >= 1:
        number_spaces = calculate_spaces(centroids[:, 0], celda_comentario, 8)
    if (len_char>=1) and (len_char <= 25):
        print ('Comentarios: OK')
    else:
        print('Comentarios: Mal')

    grupo_celda = [(celda_p1_si, celda_p1_no), (celda_p2_si, celda_p2_no), (celda_p3_si, celda_p3_no)]
    i = 1
    for si, no in grupo_celda:
        len_char_si, centroids_si = get_components(si)
        len_char_no, centroids_no = get_components(no)
        if len_char_si == 0 and len_char_no == 1:
            print('Pregunta {}: OK'.format(i))
        elif len_char_si == 1 and len_char_no == 0:
            print('Pregunta {}: OK'.format(i))
        else:
            print('Pregunta {}: Mal'.format(i))
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type = str, help = 'path for the image')
    args = parser.parse_args()

    image = args.image_path
    validacion_general(image)
