with open("dataset_word2vec.txt", 'r', encoding='utf-8') as archivo:
    contenido = archivo.readlines()
parejas = []
ventana = 4
for linea in contenido:
    limpio = linea.rstrip("\n")
    palabras = limpio.split(" ")
    for i in range(len(palabras)):
        objetivo = palabras[i]
        for j in range(1, ventana + 1):
            if i > j-1:
                parejas.append((objetivo, palabras[i-j]))
            if i < len(palabras)-j:
                parejas.append((objetivo, palabras[i+j]))
    
print(len(parejas))