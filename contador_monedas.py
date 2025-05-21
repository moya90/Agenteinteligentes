import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Colores predominantes de los billetes colombianos en formato BGR
COLORES_BILLETES = {
    2000: (255, 0, 0),    # Azul (BGR: mucho azul)
    5000: (0, 100, 200),  # Marrón
    10000: (0, 0, 255),   # Rojo
    20000: (0, 255, 0),   # Verde
    50000: (200, 0, 200), # Morado
    100000: (0, 165, 255) # Naranja
}

# Colores y diámetros de las monedas colombianas
MONEDAS = {
    50: {"color": (140, 140, 140), "diametro_mm": 17.0},   # Plateado
    100: {"color": (140, 140, 140), "diametro_mm": 20.3},  # Plateado
    200: {"color": (140, 140, 140), "diametro_mm": 22.4},  # Plateado
    500: {"color": (140, 140, 140), "diametro_mm": 23.5},  # Plateado
    1000: {"color": (120, 120, 180), "diametro_mm": 26.7}  # Bicolor (plateado/dorado)
}

def detectar_forma(imagen):
    # Convertir la imagen a formato BGR (OpenCV)
    img = np.array(imagen)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convertir a escala de grises y aplicar desenfoque
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    desenfoque = cv2.GaussianBlur(gris, (11, 11), 0)
    
    # Detectar círculos (monedas) con Hough Circle Transform
    circulos = cv2.HoughCircles(
        desenfoque,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=200
    )
    
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        # Tomar el primer círculo detectado (asumimos una moneda por imagen)
        x, y, radio = circulos[0][0]
        return "moneda", img, radio
    
    # Si no se detectan círculos, buscar formas rectangulares (billetes)
    _, umbral = cv2.threshold(gris, 240, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contorno in contornos:
        peri = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * peri, True)
        if len(aprox) == 4:  # Forma rectangular
            area = cv2.contourArea(contorno)
            if area > 10000:  # Filtrar áreas pequeñas
                return "billete", img, None
    
    return "desconocido", img, None

def obtener_color_dominante(imagen):
    # Convertir la imagen a formato BGR (OpenCV)
    img = np.array(imagen)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Reducir ruido con un desenfoque
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Aplastar la imagen a un arreglo 1D para calcular el color dominante
    pixels = img.reshape(-1, 3)
    
    # Calcular el color promedio (en BGR)
    color_promedio = np.mean(pixels, axis=0).astype(int)
    
    return color_promedio

def clasificar_objeto(tipo, color, radio=None):
    if tipo == "billete":
        colores = COLORES_BILLETES
        menor_diferencia = float('inf')
        denominacion_estimada = None
        
        for denom, color_billete in colores.items():
            diferencia = np.sqrt(sum((color - color_billete) ** 2))
            if diferencia < menor_diferencia:
                menor_diferencia = diferencia
                denominacion_estimada = denom
        return denominacion_estimada
    
    elif tipo == "moneda":
        # Convertir el radio en píxeles a un diámetro aproximado en mm
        # Asumimos una escala aproximada (calibrar según tus imágenes)
        pixels_por_mm = 5.0  # Ajustar según la resolución de tu imagen
        diametro_mm = (radio * 2) / pixels_por_mm
        
        # Clasificar basándose en color y tamaño
        menor_diferencia = float('inf')
        denominacion_estimada = None
        
        for denom, datos in MONEDAS.items():
            color_moneda = datos["color"]
            diametro_esperado = datos["diametro_mm"]
            
            # Diferencia de color
            dif_color = np.sqrt(sum((color - color_moneda) ** 2))
            # Diferencia de diámetro
            dif_diametro = abs(diametro_mm - diametro_esperado)
            
            # Combinar diferencias (dar más peso al diámetro)
            diferencia = dif_color + (dif_diametro * 10)  # Multiplicar para priorizar el tamaño
            
            if diferencia < menor_diferencia:
                menor_diferencia = diferencia
                denominacion_estimada = denom
        
        return denominacion_estimada
    
    return None

def main():
    st.title("Clasificador de Monedas y Billetes Colombianos por Color y Tamaño")
    st.write("Sube una imagen de una moneda o billete colombiano y te diré su denominación basándome en el color y la forma.")
    
    # Subir imagen
    archivo_subido = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
    
    if archivo_subido is not None:
        try:
            # Leer y mostrar la imagen subida
            imagen = Image.open(archivo_subido)
            st.image(imagen, caption="Imagen Subida", use_column_width=True)
            
            # Detectar si es moneda o billete
            with st.spinner("Analizando forma, color y tamaño..."):
                tipo, img, radio = detectar_forma(imagen)
                if tipo == "desconocido":
                    st.error("No se pudo determinar si es una moneda o un billete.")
                    return
                
                color_dominante = obtener_color_dominante(imagen)
                denominacion = clasificar_objeto(tipo, color_dominante, radio)
            
            # Mostrar resultados
            st.subheader("Resultado")
            if denominacion:
                tipo_texto = "moneda" if tipo == "moneda" else "billete"
                st.write(f"Este {tipo_texto} parece ser de **{denominacion:,} COP**.")
                st.write(f"Color dominante detectado (BGR): {color_dominante}")
            else:
                st.write("No se pudo determinar la denominación.")
        
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

if __name__ == "__main__":
    main()