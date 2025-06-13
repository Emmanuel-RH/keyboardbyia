# 🖐️ Teclado Virtual Controlado con el Dedo Índice

Este proyecto experimental implementa un teclado virtual completamente funcional que puede ser controlado **únicamente con el dedo índice**, detectado a través de la cámara web. Utilizando visión por computadora (MediaPipe) y técnicas simples de aprendizaje automático para autocompletado, esta interfaz alternativa elimina la necesidad de dispositivos físicos, abriendo posibilidades para entornos accesibles o sin contacto.

## 🚀 Características Principales

- ✅ Escritura de texto completa con solo mover el dedo índice
- 🔤 Autocompletado dinámico de palabras comunes
- ⌫ Funciones de borrado individual y total
- ⌨️ Interfaz de teclado QWERTY adaptada
- 🧠 Detección precisa mediante seguimiento en tiempo real con MediaPipe

## 🎯 Casos de Uso

- Personas con movilidad reducida
- Interacción sin contacto (entornos clínicos o estériles)
- Proyectos educativos de visión computacional e interfaces hombre-máquina
- Prototipado de sistemas de accesibilidad

---

## 🛠️ Requisitos

- Python 3.8 o superior  
- Sistema operativo Windows (por uso de `winsound`)  
- Cámara web activa y buena iluminación
 

### 🔧 Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/teclado_virtual_mano.git
   cd teclado_virtual_mano
   posteriormente abrir en VisualStudioCode

## 📝 Notas Técnicas

- El sistema rastrea la posición del dedo índice (landmark 8) y activa teclas tras una breve permanencia.
- La tecla "0" ejecuta el autocompletado cuando hay una sugerencia disponible.
- El teclado incluye funciones especiales como espacio (␣), borrar (←) y eliminar todo (⎚).
  
## 📦 Dependencias Principales

- opencv-python
- mediapipe
- numpy
- difflib (estándar en Python)
- winsound (solo disponible en Windows)

Ejecuta el siguiente comando en tu entorno virtual:
  ```bash
  pip install opencv-python mediapipe numpy



