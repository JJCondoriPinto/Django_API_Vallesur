from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import os


def recortarDatos(original, p1, p2):
    try:
        # Crear una máscara con las dimensiones de la imagen original
        mask = np.zeros(original.shape[:2], dtype='uint8')

        # Dibujar el rectángulo en la máscara
        cv2.rectangle(mask, p1, p2, 255, -1)

        # Aplicar la máscara a la región de interés de la imagen umbralizada
        roi = original[p1[1]:p2[1], p1[0]:p2[0]]
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask[p1[1]:p2[1], p1[0]:p2[0]])

        return roi_masked

    except Exception as e:
        raise Exception(f'Error en la función de recorte de datos: {str(e)}')


def cortarDatosNum_DNI(original):
    p1 = (882, 69)
    p2 = (1480-371, 190-40)
    return recortarDatos(original, p1, p2)

def cortarDatosApe_p(original):
    p1 = (387, 129)
    p2 = (940 - 371, 318 - 120)
    return recortarDatos(original, p1, p2)


def cortarDatosApe_m(original):
    p1 = (387, 198)
    p2 = (945 - 371, 310 - 40)
    return recortarDatos(original, p1, p2)


def cortarDatosPrenombre(original):
    p1 = (387, 270)
    p2 = (970 - 371, 381 - 40)
    return recortarDatos(original, p1, p2)


def cortarDatosfech_nac(original):
    p1 = (388, 340)
    p2 = (881 - 371, 438 - 40)
    return recortarDatos(original, p1, p2)


def cortarDatosDepartamento(original):
    p1 = (175, 300)
    p2 = (745 - 371, 401 - 40)
    return recortarDatos(original, p1, p2)


def cortarDatosProvincia(original):
    p1 = (373, 300)
    p2 = (1000 - 371, 401 - 40)
    return recortarDatos(original, p1, p2)


def cortarDatosDistrito(original):
    p1 = (650, 300)
    p2 = (1220 - 371, 401 - 40)
    return recortarDatos(original, p1, p2)


def cortarDatosDireccion(original):
    p1 = (175, 354)
    p2 = (970 - 371, 475 - 40)
    return recortarDatos(original, p1, p2)


#_________________________________________________________________________________________________________________



def mostrar_imagen_num_dni(imageFormat):
    try:
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(imageFormat, cv2.COLOR_BGR2GRAY)
        # Aplicar un filtro de desenfoque para reducir el ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Aplicar umbral adaptativo utilizando el método de Otsu
        umbral = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 7)

        # Convertir la imagen a formato de PIL
        pil_image = Image.fromarray(umbral)

        # Aplicar la secuencia de operaciones del comando PIL
        pil_image = pil_image.convert('L').resize([3 * _ for _ in pil_image.size], Image.BICUBIC)
        pil_image = pil_image.point(lambda p: p > 75 and p + 100)

        # Convertir la imagen de PIL a formato de OpenCV
        umbral = np.array(pil_image)

        return umbral
    except Exception as e:
        raise Exception('Error en la función "mostrar_imagen_num_dni": {}'.format(str(e)))


def mostrar_imagen_text_dni(imageFormat):
    try:
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(imageFormat, cv2.COLOR_BGR2GRAY)
        # Aplicar un filtro de desenfoque para reducir el ruido
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Aplicar umbral adaptativo utilizando el método de Otsu
        umbral = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 9)

        # Convertir la imagen a formato de PIL
        pil_image = Image.fromarray(umbral)

        # Aplicar la secuencia de operaciones del comando PIL
        pil_image = pil_image.convert('L').resize([3 * _ for _ in pil_image.size], Image.BICUBIC)
        pil_image = pil_image.point(lambda p: p > 75 and p + 100)

        # Convertir la imagen de PIL a formato de OpenCV
        umbral = np.array(pil_image)

        return umbral
    except Exception as e:
        raise Exception('Error en la función "mostrar_imagen_text_dni": {}'.format(str(e)))


def mostrar_imagen_date_dni(imageFormat):
    try:
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(imageFormat, cv2.COLOR_BGR2GRAY)
        # Aplicar un filtro de desenfoque para reducir el ruido
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Aplicar umbral adaptativo utilizando el método de Otsu
        umbral = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 43, 10)

        # Convertir la imagen a formato de PIL
        pil_image = Image.fromarray(umbral)

        # Aplicar la secuencia de operaciones del comando PIL
        pil_image = pil_image.convert('L').resize([3 * _ for _ in pil_image.size], Image.BICUBIC)
        pil_image = pil_image.point(lambda p: p > 75 and p + 100)

        # Convertir la imagen de PIL a formato de OpenCV
        umbral = np.array(pil_image)

        return umbral
    except Exception as e:
        raise Exception('Error en la función "mostrar_imagen_date_dni": {}'.format(str(e)))


#_________________________________________________________________________________________________________________


def obtener_datos_num_dni(imageText):
    try:
        # Obtener el texto de la imagen utilizando PyTesseract
        options = '-c tessedit_char_whitelist=0123456789'
        texto = pytesseract.image_to_string(imageText, config=options)

        return texto
    
    except Exception as e:
        raise Exception('Error en la función "obtener_datos_num_dni": {}'.format(str(e)))


def obtener_datos_text_dni(imageText):
    try:
        # Obtener el texto de la imagen utilizando PyTesseract
        options2 = '-c tessedit_char_whitelist= ABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ.123456789'
        texto = pytesseract.image_to_string(imageText, config=options2)

        return texto
    
    except Exception as e:
        raise Exception('Error en la función "obtener_datos_text_dni": {}'.format(str(e)))


def obtener_datos_date_dni(imageText):
    try:
        # Obtener el texto de la imagen utilizando PyTesseract
        options = '-c tessedit_char_whitelist= 0123456789'
        texto = pytesseract.image_to_string(imageText, config=options)

        return texto
    
    except Exception as e:
        raise Exception('Error en la función "obtener_datos_date_dni": {}'.format(str(e)))

#_________________________________________________________________________________________________________________

def convertir_imagen_delantera(imagenConvert):
        try:
        #if request.method == 'POST' and request.FILES.get('imagen'):

            pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

            #imagen = request.FILES['imagen']
            #imagen_dni = cv2.imread(imagenConvert)
            
            imagen_dni = imagenConvert
            #imagen_dni = cv2.imdecode(np.frombuffer(imagen.read(), np.uint8), cv2.IMREAD_COLOR)
            
            imageShorted_dniNum = cortarDatosNum_DNI(imagen_dni)
            imageShorted_apePat = cortarDatosApe_p(imagen_dni)
            imageShorted_apeMat = cortarDatosApe_m(imagen_dni)
            imageShorted_nombres = cortarDatosPrenombre(imagen_dni)
            imageShorted_fechNac = cortarDatosfech_nac(imagen_dni)

            imagenConvert_dni_dniNum = mostrar_imagen_num_dni(imageShorted_dniNum)
            imagenConvert_dni_apePat = mostrar_imagen_text_dni(imageShorted_apePat)
            imagenConvert_dni_apeMat = mostrar_imagen_text_dni(imageShorted_apeMat)
            imagenConvert_dni_nombres = mostrar_imagen_text_dni(imageShorted_nombres)
            imagenConvert_dni_fechNac = mostrar_imagen_date_dni(imageShorted_fechNac)

            datos_dni_dniNum = obtener_datos_num_dni(imagenConvert_dni_dniNum)
            datos_dni_dni_apePat = obtener_datos_text_dni(imagenConvert_dni_apePat)
            datos_dni_dni_apeMat = obtener_datos_text_dni(imagenConvert_dni_apeMat)
            datos_dni_nombres = obtener_datos_text_dni(imagenConvert_dni_nombres)
            datos_dni_fechNac = obtener_datos_date_dni(imagenConvert_dni_fechNac)

            linea_dniNum = datos_dni_dniNum.splitlines()  # Dividir la cadena en líneas
            linea_dni_apePat = datos_dni_dni_apePat.splitlines()  # Dividir la cadena en líneas
            linea_dni_apeMat = datos_dni_dni_apeMat.splitlines()  # Dividir la cadena en líneas
            linea_nombres = datos_dni_nombres.splitlines()  # Dividir la cadena en líneas
            linea_fechNac = datos_dni_fechNac.splitlines()  # Dividir la cadena en líneas

            dni_clean = linea_dniNum[1] if len(linea_dniNum) >= 2 else datos_dni_dniNum
            apePat_clean = linea_dni_apePat[1] if len(linea_dni_apePat) >= 2 else datos_dni_dni_apePat
            apeMat_clean = linea_dni_apeMat[1] if len(linea_dni_apeMat) >= 2 else datos_dni_dni_apeMat
            nombres_clean = linea_nombres[1] if len(linea_nombres) >= 2 else datos_dni_nombres
            fecNac_clean = linea_fechNac[1] if len(linea_fechNac) >= 2 else datos_dni_fechNac

                #pattern = r'[$%0\n\t]'

                #JSON_apePat = re.sub(pattern, '', apePat_clean)
            
            resultado_json = {
                    "dniNum": dni_clean,
                    "apePat": apePat_clean,
                    "apeMat": apeMat_clean,
                    "nombres": nombres_clean,
                    "fechNac": fecNac_clean,
            }

            return resultado_json
        

        #return JsonResponse({'success': 400, 'error': 'Se esperaba una imagen en el POST.'})
        except Exception as e:
            raise Exception('Se esperaba una imagen en el post": {}'.format(str(e)))

#_________________________________________________________________________________________________________________

def convertir_imagen_trasera(imagenConvert):
        try:
        #if request.method == 'POST' and request.FILES.get('imagen'):
        
            pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

            #imagen = request.FILES['imagen']
            #imagen_dni = cv2.imread(imagenConvert)

            imagen_dni = imagenConvert
            
            #imagen_dni = cv2.imdecode(np.frombuffer(imagen.read(), np.uint8), cv2.IMREAD_COLOR)
            
            imageShorted_departamento = cortarDatosDepartamento(imagen_dni)
            imageShorted_provincia = cortarDatosProvincia(imagen_dni)
            imageShorted_distrito = cortarDatosDistrito(imagen_dni)
            imageShorted_direccion = cortarDatosDireccion(imagen_dni)

            imagenConvert_dni_departamento = mostrar_imagen_text_dni(imageShorted_departamento)
            imagenConvert_dni_provincia = mostrar_imagen_text_dni(imageShorted_provincia)
            imagenConvert_dni_distrito = mostrar_imagen_text_dni(imageShorted_distrito)
            imagenConvert_dni_direccion = mostrar_imagen_text_dni(imageShorted_direccion)
        
            datos_dni_departamento = obtener_datos_text_dni(imagenConvert_dni_departamento)
            datos_dni_provincia = obtener_datos_text_dni(imagenConvert_dni_provincia)
            datos_dni_distrito = obtener_datos_text_dni(imagenConvert_dni_distrito)
            datos_dni_direccion = obtener_datos_text_dni(imagenConvert_dni_direccion)

            linea_departamento = datos_dni_departamento.splitlines()  # Dividir la cadena en líneas
            linea_provincia = datos_dni_provincia.splitlines()  # Dividir la cadena en líneas
            linea_distrito = datos_dni_distrito.splitlines()  # Dividir la cadena en líneas
            linea_direccion = datos_dni_direccion.splitlines()  # Dividir la cadena en líneas

            departamento_clean = linea_departamento[1] if len(linea_departamento) >= 2 else datos_dni_departamento
            provincia_clean = linea_provincia[1] if len(linea_provincia) >= 2 else datos_dni_provincia
            distrito_clean = linea_distrito[1] if len(linea_distrito) >= 2 else datos_dni_distrito
            direccion_clean = linea_direccion[1] if len(linea_direccion) >= 2 else datos_dni_direccion
            
            resultado_json = {
                    "dep": departamento_clean,
                    "prov": provincia_clean,
                    "dist": distrito_clean,
                    "dir": direccion_clean,
                }

            return resultado_json

        except Exception as e:
            raise Exception('Se esperaba una imagen en el post": {}'.format(str(e)))

@csrf_exempt
def convertir_imagen_endpoint(request):
    try:
        if request.method == 'POST' and request.FILES.get('image_bytes1') and request.FILES.get('image_bytes2'):

            imagen1 = request.FILES['image_bytes1']
            imagen2 = request.FILES['image_bytes2']

        # if request.method == 'POST':

            
        #     image_bytes1 = request.POST.get('image_bytes1')
        #     if image_bytes1:
        #         # Convertir la lista de bits en un arreglo de NumPy
        #         image_bytes1 = [int(x) for x in image_bytes1.split(',')]
        #         image_bytes1 = bytes(image_bytes1)

        #         decoded_image1 = cv2.imdecode(np.frombuffer(image_bytes1, np.uint8), -1)   
            
            
        #     image_bytes2 = request.POST.get('image_bytes2')
        #     if image_bytes2:
        #         # Convertir la lista de bits en un arreglo de NumPy
        #         image_bytes2 = [int(x) for x in image_bytes2.split(',')]
        #         image_bytes2 = bytes(image_bytes2)

        #         decoded_image2 = cv2.imdecode(np.frombuffer(image_bytes2, np.uint8), -1) 

            # Leer la imagen utilizando OpenCV
            imagen1_cv2 = cv2.imdecode(np.fromstring(imagen1.read(), np.uint8), cv2.IMREAD_COLOR)
            imagen2_cv2 = cv2.imdecode(np.fromstring(imagen2.read(), np.uint8), cv2.IMREAD_COLOR)
            

            resultadoImage1 = convertir_imagen_delantera(imagen1_cv2)
            resultadoImage2 = convertir_imagen_trasera(imagen2_cv2)

            resultado_concatenado = {}
            resultado_concatenado.update(resultadoImage1)
            resultado_concatenado.update(resultadoImage2)

            response_data = {
                'status': 200,
                'message': 'Extracción de datos exitosa',
                'data': resultado_concatenado
            }

            return JsonResponse(response_data)

        return JsonResponse({'status': 400, 'error': 'Se esperaba una imagen en el POST.'})

    except Exception as e:
        return JsonResponse({'status': 404, 'error': str(e)})

#_________________________________________________________________________________________________________________

def convertir_imagen_view(request):
    return render(request, 'formulario.html')
