import cv2
import mediapipe as mp
import numpy as np
import time
import winsound  # Para el sonido en Windows

# --- NUEVO: Librerías para el predictor neuronal ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Predictor neuronal LSTM ---
class NeuralWordPredictor:
    """
    Predictor de palabras basado en una Red Neuronal LSTM.
    """
    def __init__(self, corpus_path):
        print("Inicializando predictor neuronal...")
        self.model = None
        self.char_to_int = {}
        self.int_to_char = {}
        self.max_len = 0
        self.vocab_size = 0
        self._train_model(corpus_path)

    def _train_model(self, corpus_path):
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                words = [line.strip().lower() + ' ' for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Advertencia: '{corpus_path}' no encontrado. Usando corpus por defecto para el predictor.")
            words = ["hola ", "mundo ", "ayuda ", "proyecto ", "inteligencia ", "artificial ", "teclado ", "mano "]
        text = "".join(words)
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_int = {c: i for i, c in enumerate(chars)}
        self.int_to_char = {i: c for i, c in enumerate(chars)}
        sequences = []
        for word in words:
            for i in range(1, len(word)):
                sequences.append(word[:i+1])
        self.max_len = max([len(seq) for seq in sequences])
        encoded_sequences = [[self.char_to_int[char] for char in seq] for seq in sequences]
        padded_sequences = pad_sequences(encoded_sequences, maxlen=self.max_len, padding='pre')
        X, y = padded_sequences[:, :-1], padded_sequences[:, -1]
        y = to_categorical(y, num_classes=self.vocab_size)
        print("Entrenando modelo de predicción de palabras (esto puede tardar unos segundos)...")
        self.model = Sequential([
            Embedding(self.vocab_size, 50, input_length=self.max_len-1),
            LSTM(100),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X, y, epochs=20, verbose=0)
        print("Modelo de predicción entrenado y listo.")

    def predict(self, prefix, max_sugg=1):
        if not self.model or not prefix:
            return []
        original_prefix = prefix.lower()
        for _ in range(self.max_len - len(original_prefix)):
            encoded = [self.char_to_int[char] for char in original_prefix if char in self.char_to_int]
            if not encoded:
                break
            padded = pad_sequences([encoded], maxlen=self.max_len-1, padding='pre')
            pred_probs = self.model.predict(padded, verbose=0)[0]
            pred_index = np.argmax(pred_probs)
            next_char = self.int_to_char.get(pred_index, '')
            if next_char == ' ':
                break
            original_prefix += next_char
        return [original_prefix] if len(original_prefix) > len(prefix.lower()) else []

# --- Inicializar MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Teclas del teclado (agrega "⎚" como tecla de borrado total)
teclas = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM←␣⎚")
]

# Etiquetas para teclas especiales
etiquetas = {
    "␣": "SPACE",
    "←": "DEL",
    "⎚": "DEL ALL"
}

def calcular_anchos_teclas():
    anchos = []
    for fila in teclas:
        fila_anchos = []
        for tecla in fila:
            txt = etiquetas.get(tecla, tecla)
            ancho = max(60, 20 * len(txt) + 20)
            fila_anchos.append(ancho)
        anchos.append(fila_anchos)
    return anchos

anchos_teclas = calcular_anchos_teclas()
alto_celda = 60
texto_escrito = ""
seleccion_previa = None
tiempo_seleccion = 0

# --- INICIALIZA el predictor neuronal ---
predictor = NeuralWordPredictor("corpus.txt")  # Cambia el nombre si tienes otro corpus

def obtener_sugerencia(texto):
    if not texto.strip():
        return ""
    ult_palabra = texto.split(" ")[-1]
    sugerencias = predictor.predict(ult_palabra)
    if sugerencias:
        sugerencia = sugerencias[0][len(ult_palabra):]
        return sugerencia.upper()
    return ""

def dibujar_teclado(frame, seleccion):
    h, w, _ = frame.shape
    y_inicio = h - len(teclas) * alto_celda - 50
    for i, fila in enumerate(teclas):
        fila_anchos = anchos_teclas[i]
        ancho_fila = sum(fila_anchos)
        x_inicio = (w - ancho_fila) // 2
        x = x_inicio
        for j, tecla in enumerate(fila):
            ancho = fila_anchos[j]
            color = (0, 255, 0) if (i, j) == seleccion else (50, 50, 50)
            cv2.rectangle(frame, (x, y_inicio + i * alto_celda), (x + ancho, y_inicio + (i + 1) * alto_celda), color, -1)
            txt = etiquetas.get(tecla, tecla)
            (text_w, text_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = x + (ancho - text_w) // 2
            text_y = y_inicio + i * alto_celda + (alto_celda + text_h) // 2
            cv2.putText(frame, txt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            x += ancho

def obtener_tecla_enfocada(x, y, w, h):
    y_inicio = h - len(teclas) * alto_celda - 50
    for i, fila in enumerate(teclas):
        fila_anchos = anchos_teclas[i]
        ancho_fila = sum(fila_anchos)
        x_inicio = (w - ancho_fila) // 2
        x_celda = x_inicio
        for j, ancho in enumerate(fila_anchos):
            if (x_celda <= x < x_celda + ancho) and (y_inicio + i * alto_celda <= y < y_inicio + (i + 1) * alto_celda):
                return (i, j)
            x_celda += ancho
    return None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

sugerencia = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)

    seleccion = None

    # --- SUGERENCIA usando el predictor neuronal ---
    sugerencia = obtener_sugerencia(texto_escrito)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            seleccion = obtener_tecla_enfocada(x, y, w, h)

            if seleccion:
                if seleccion == seleccion_previa:
                    if time.time() - tiempo_seleccion > 1.0:
                        letra = teclas[seleccion[0]][seleccion[1]]
                        if letra == "←":
                            texto_escrito = texto_escrito[:-1]
                        elif letra == "␣":
                            texto_escrito += " "
                        elif letra == "⎚":
                            texto_escrito = ""
                        elif letra == "0" and sugerencia:
                            texto_escrito += sugerencia
                        else:
                            texto_escrito += letra
                        winsound.Beep(1000, 400)
                        tiempo_seleccion = time.time()
                else:
                    seleccion_previa = seleccion
                    tiempo_seleccion = time.time()

    cuadro_x, cuadro_y, cuadro_w, cuadro_h = 30, 30, w - 60, 70
    cv2.rectangle(frame, (cuadro_x, cuadro_y), (cuadro_x + cuadro_w, cuadro_y + cuadro_h), (30, 30, 30), -1)
    cv2.rectangle(frame, (cuadro_x, cuadro_y), (cuadro_x + cuadro_w, cuadro_y + cuadro_h), (0, 255, 255), 2)
    texto_mostrar = texto_escrito
    while True:
        (text_w, _), _ = cv2.getTextSize(texto_mostrar, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        if text_w < cuadro_w - 20 or len(texto_mostrar) == 0:
            break
        texto_mostrar = texto_mostrar[1:]
    cv2.putText(frame, texto_mostrar, (cuadro_x + 10, cuadro_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    if sugerencia:
        (text_w, _), _ = cv2.getTextSize(texto_mostrar, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(frame, sugerencia, (cuadro_x + 10 + text_w, cuadro_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)

    dibujar_teclado(frame, seleccion)
    cv2.imshow("Teclado con Mano", frame)
    cv2.resizeWindow("Teclado con Mano", 1280, 720)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()