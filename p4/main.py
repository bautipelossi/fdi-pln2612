# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "beautifulsoup4",
# ]
# ///

import os
import time
import unicodedata
from bs4 import BeautifulSoup

# ==========================================
# MÓDULO 1: PREPROCESAMIENTO Y LIMPIEZA
# ==========================================
def limpiar_texto_para_busqueda(texto):
    """
    Normaliza el texto: a minúsculas y sin tildes.
    Vital para que la búsqueda sea tolerante a errores humanos.
    """
    texto = texto.lower()
    # Descompone los caracteres (ej: á -> a + ´) y elimina las marcas diacríticas
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    return texto

def extraer_pasajes(ruta_archivo):
    """ Extrae el HTML y lo convierte en una lista estructurada de párrafos """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"❌ Error crítico: Falta el archivo {ruta_archivo}")

    print("⚙️ [1/3] Extrayendo y limpiando el HTML (esto puede tomar unos segundos)...")
    
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        sopa = BeautifulSoup(archivo, 'html.parser')
        
    for etiqueta in sopa(["script", "style"]):
        etiqueta.decompose() 
        
    pasajes = []
    for bloque in sopa.find_all(['p', 'h1', 'h2', 'h3', 'h4']):
        texto_limpio = bloque.get_text(separator=' ', strip=True)
        if len(texto_limpio) > 15: 
            pasajes.append(texto_limpio)
            
    return pasajes

# ==========================================
# MÓDULO 2: MOTOR DE BÚSQUEDA
# ==========================================
def buscar_en_texto(pasajes, consulta):
    """
    Busca una frase o palabra usando normalización semántica básica.
    Devuelve los pasajes exactos donde ocurre.
    """
    print(f"🔍 [2/3] Buscando la frase: '{consulta}'...")
    consulta_normalizada = limpiar_texto_para_busqueda(consulta)
    resultados = []
    
    for pasaje in pasajes:
        pasaje_normalizado = limpiar_texto_para_busqueda(pasaje)
        if consulta_normalizada in pasaje_normalizado:
            resultados.append(pasaje)
            
    return resultados

# ==========================================
# ENTRY POINT (PUNTO DE EJECUCIÓN)
# ==========================================
if __name__ == '__main__':
    archivo_quijote = '2000-h.htm'
    
    # FRASE A BUSCAR (Puedes cambiarla para probar)
    frase_prueba = "En un lugar de la Mancha"
    
    try:
        inicio_tiempo = time.time()
        
        # 1. Pipeline de datos
        lista_de_pasajes = extraer_pasajes(archivo_quijote)
        
        # 2. Pipeline de búsqueda
        pasajes_encontrados = buscar_en_texto(lista_de_pasajes, frase_prueba)
        
        # 3. Métricas y Resultados
        tiempo_total = time.time() - inicio_tiempo
        print(f"✅ [3/3] Búsqueda completada en {tiempo_total:.2f} segundos.")
        print("-" * 60)
        print(f"🎯 Se encontraron {len(pasajes_encontrados)} pasaje(s) con tu consulta.")
        print("-" * 60)
        
        # Mostrar los primeros 3 resultados para no saturar la terminal
        for i, pasaje in enumerate(pasajes_encontrados[:3], 1):
            print(f"📖 RESULTADO {i}:\n{pasaje}\n")
            
    except Exception as e:
        print(f"🚨 Error en la arquitectura: {e}")
