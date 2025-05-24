import numpy as np
import math
from scipy.optimize import linear_sum_assignment

def sbox_to_prob_matrix_np(sbox_list, n, m):
  """
  Convierte una S-box estándar (representada como una lista de enteros)
  a su forma de matriz de probabilidad determinística, usando NumPy.

  Args:
    sbox_list: Una lista de enteros donde sbox_list[i] es la salida
               correspondiente a la entrada i.
    n: El número de bits de entrada de la S-box.
    m: El número de bits de salida de la S-box.

  Returns:
    Un array de NumPy de tipo float64 con forma (2^n, 2^m).
    El elemento [i][j] será 1.0 si la S-box mapea la entrada i
    a la salida j, y 0.0 en caso contrario.

  Raises:
    ValueError: Si la longitud de la lista no coincide con 2^n o
                si contiene valores fuera del rango [0, 2^m-1].
  """

  num_inputs = 2**n
  num_outputs = 2**m

  # --- Validación de Entradas (igual que antes) ---
  if len(sbox_list) != num_inputs:
    raise ValueError(
        f"La longitud de la S-box ({len(sbox_list)}) no coincide con 2^n "
        f"({num_inputs})."
    )

  sbox_np = np.array(sbox_list) # Convertimos a array NumPy para la validación

  if not np.all((sbox_np >= 0) & (sbox_np < num_outputs)):
      raise ValueError(
          f"La S-box contiene valores de salida fuera del rango "
          f"[0, {num_outputs-1}]."
      )

  # --- Creación de la Matriz con NumPy ---
  # Inicializamos una matriz de 2^n x 2^m con ceros (0.0).
  # Usamos float64 para mayor precisión en cálculos futuros.
  prob_matrix = np.zeros((num_inputs, num_outputs), dtype=np.float64)

  # --- Relleno de la Matriz con Indexación Avanzada ---
  # Creamos un array con los índices de las filas (0, 1, 2, ..., 2^n - 1)
  row_indices = np.arange(num_inputs)

  # Los índices de las columnas son directamente los valores de la S-box
  col_indices = sbox_np

  # Usamos la indexación avanzada de NumPy para poner 1.0 en las
  # posiciones (i, S(i)) de forma vectorizada.
  # Esto es mucho más rápido que un bucle for en Python para arrays grandes.
  prob_matrix[row_indices, col_indices] = 1.0

  return prob_matrix


def prob_matrix_to_sbox(prob_matrix):
  """
  Convierte una matriz de probabilidad P (NumPy array o lista de listas)
  a una S-box estándar (lista de enteros), eligiendo la salida
  con la mayor probabilidad para cada entrada.

  Args:
    prob_matrix: Un array de NumPy o una lista de listas representando
                 la matriz de probabilidad P (tamaño 2^n x 2^m).

  Returns:
    Una lista de enteros representando la S-box. sbox_list[i] será
    la salida 'j' que maximiza P_ij.
  """

  # Asegurarnos de que trabajamos con un array de NumPy
  p_matrix = np.array(prob_matrix)

  # --- Verificación de Forma (Opcional pero recomendable) ---
  if p_matrix.ndim != 2:
      raise ValueError("La entrada debe ser una matriz 2D.")

  # --- Encontrar la Salida más Probable ---
  # np.argmax(axis=1) encuentra el *índice* del valor máximo
  # a lo largo de cada fila (axis=1).
  # Este índice corresponde directamente al valor de salida 'j'
  # de la S-box.
  sbox_array = np.argmax(p_matrix, axis=1)

  # --- Convertir a Lista y Devolver ---
  # Convertimos el array de NumPy resultante a una lista estándar de Python.
  return sbox_array.tolist()


def calculate_ddt_prob(prob_matrix):
  """
  Calcula la Tabla de Distribución Diferencial (DDT) probabilística
  a partir de una matriz de probabilidad P.

  Args:
    prob_matrix: Un array de NumPy representando la matriz P (tamaño 2^n x 2^m).

  Returns:
    Un array de NumPy representando la DDT (tamaño 2^n x 2^m).
  """

  # Asegurarnos de que es un array NumPy
  P = np.array(prob_matrix, dtype=np.float64)

  # Obtener dimensiones
  num_inputs, num_outputs = P.shape
  n = int(np.log2(num_inputs))
  m = int(np.log2(num_outputs))

  # Inicializar la DDT con ceros
  ddt = np.zeros((num_inputs, num_outputs), dtype=np.float64)

  # Pre-calcular los índices de entrada y salida
  inputs_idx = np.arange(num_inputs)
  outputs_idx = np.arange(num_outputs)

  # Iterar sobre todas las posibles diferencias de entrada (delta_x)
  for delta_x in range(num_inputs):
    # Calcular P_{x ^ delta_x, y} para todos los x, y
    # Esto se hace reordenando las filas de P según x ^ delta_x
    P_shifted_x = P[np.bitwise_xor(inputs_idx, delta_x), :]

    # Iterar sobre todas las posibles diferencias de salida (delta_y)
    for delta_y in range(num_outputs):
      # Calcular P_{x ^ delta_x, y ^ delta_y} para todos los x, y
      # Esto se hace reordenando las columnas de P_shifted_x
      # según y ^ delta_y
      P_shifted_xy = P_shifted_x[:, np.bitwise_xor(outputs_idx, delta_y)]

      # Calcular DDT(delta_x, delta_y)
      # Multiplicamos P y P_shifted_xy elemento a elemento
      # (P_{x,y} * P_{x^dx, y^dy}) y luego sumamos todos los resultados.
      # Esta es la suma interna vectorizada.
      ddt[delta_x, delta_y] = np.sum(P * P_shifted_xy)

  return ddt

def fwht(a):
  """Aplica la Transformada Rápida de Walsh-Hadamard 1D (in-place)."""
  n = len(a)
  h = 1
  while h < n:
      for i in range(0, n, h * 2):
          for j in range(i, i + h):
              x = a[j]
              y = a[j + h]
              a[j] = x + y
              a[j + h] = x - y
      h *= 2
  return a

def fwht2d(matrix):
  """Aplica la Transformada Rápida de Walsh-Hadamard 2D."""
  # Copiamos para no modificar la original
  temp = matrix.copy()
  # Aplicar a filas
  np.apply_along_axis(fwht, 1, temp)
  # Aplicar a columnas
  np.apply_along_axis(fwht, 0, temp)
  return temp

def ifwht2d(matrix):
  """Aplica la Inversa de la FWHT 2D."""
  h, w = matrix.shape
  n_total = h * w
  # La FWHT es su propia inversa, solo se necesita escalar
  return fwht2d(matrix) / n_total

def project_to_simplex(P):
    """Proyecta cada fila de P al simplex de probabilidad."""
    # 1. Asegurar no negatividad
    P_proj = np.maximum(P, 0)
    # 2. Normalizar cada fila para que sume 1
    row_sums = np.sum(P_proj, axis=1, keepdims=True)
    # Evitar división por cero (aunque improbable con gradiente)
    row_sums[row_sums == 0] = 1.0
    P_proj = P_proj / row_sums
    return P_proj


def calculate_variance_gradient(P):
    """Calcula el gradiente de la varianza de la DDT respecto a P."""
    num_inputs, num_outputs = P.shape
    
    # 1. Calcular DDT
    ddt = calculate_ddt_prob(P)
    
    # 2. Calcular W = (DDT - mu), excluyendo delta_x = 0
    ddt_relevant = ddt[1:, :]
    mu = np.mean(ddt_relevant)
    W = ddt - mu
    W[0, :] = 0.0 # Ignoramos delta_x = 0 en el coste
    
    # 3. Calcular N_DDT (número de entradas relevantes)
    N_DDT = ddt_relevant.size
    
    # 4. Calcular el término principal del gradiente usando FWHT
    # G = FWHT_inv(FWHT(W) * FWHT(P))
    # El '*' aquí es XOR-correlación, que se vuelve '.' en el dominio WHT.
    G = ifwht2d(fwht2d(W) * fwht2d(P)) # '*' es mult. elemento a elemento

    # 5. Calcular el gradiente final
    # Derivada de Var(X) = Derivada(E[X^2] - E[X]^2)
    # Simplifica a (4 / N_DDT) * G para la varianza
    gradient = (4.0 / N_DDT) * G
    
    return gradient


def gd_minimize_variance(initial_prob_matrix, iterations, learning_rate, penalty_lambda):
    """
    Minimiza la Varianza + Penalización (P*(1-P)) usando Descenso de Gradiente.

    Args:
      initial_prob_matrix: La matriz P inicial (NumPy array).
      iterations: El número de pasos de optimización.
      learning_rate: La tasa de aprendizaje (eta).
      penalty_lambda: El peso (lambda) para la penalización P*(1-P).

    Returns:
      La matriz P optimizada.
    """
    P = initial_prob_matrix.copy()
    num_inputs, num_outputs = P.shape

    print(f"Iniciando GD con Penalización. Lambda: {penalty_lambda}")
    print(f"Iteraciones: {iterations}, Tasa de Aprendizaje: {learning_rate}")

    for i in range(iterations):
        # 1. Calcular gradiente de la varianza DDT (grad1)
        grad1 = calculate_variance_gradient(P)

        # 2. Calcular gradiente de la penalización (grad2)
        # grad2 = d/dP (P - P^2) = 1 - 2*P
        grad2 = 1.0 - 2.0 * P

        # 3. Calcular gradiente total
        total_grad = grad1 + penalty_lambda * grad2

        # 4. Actualizar P
        P = P - learning_rate * total_grad

        # 5. Proyectar P de vuelta al espacio válido
        P = project_to_simplex(P)

        # 6. (Opcional) Imprimir coste cada N iteraciones
        if (i + 1) % 200 == 0 or i == 0:
            ddt = calculate_ddt_prob(P)
            variance = np.var(ddt[1:, :])
            diff_uniformity = np.max(ddt[1:, :])
            # Calculamos el valor de la penalización
            penalty_cost = np.sum(P * (1.0 - P))
            total_cost = variance + penalty_lambda * penalty_cost
            print(f"Iter {i+1}/{iterations} - Coste Total: {total_cost:.6f} "
                  f"(Var: {variance:.6f}, Pen: {penalty_cost:.2f}) - "
                  f"Unif. Dif.: {diff_uniformity:.4f}")

    print("Descenso de Gradiente con Penalización completado.")
    return P