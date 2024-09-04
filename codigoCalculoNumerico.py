import numpy as np
from sympy import symbols, simplify

# Funções para resolver sistemas lineares

def jacobi(A, b, x0, tol, max_iterations):
    """
    Método de Jacobi para resolver sistemas lineares.
    
    Args:
        A: Matriz dos coeficientes.
        b: Vetor dos termos independentes.
        x0: Vetor inicial.
        tol: Tolerância para convergência.
        max_iterations: Número máximo de iterações.

    Returns:
        x_new: Vetor solução.
        k + 1: Número de iterações realizadas.
    """
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum = 0
            for j in range(n):
                if i != j:
                    sum += A[i, j] * x[j]
            x_new[i] = (b[i] - sum) / A[i, i]
        
        print(f'Iteração {k + 1}: {x_new}')  # Exibindo a iteração
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    
    raise Exception(f'Não convergiu após {max_iterations} iterações')

def gauss_seidel(A, b, x0, tol, max_iterations):
    """
    Método de Gauss-Seidel para resolver sistemas lineares.
    
    Args:
        A: Matriz dos coeficientes.
        b: Vetor dos termos independentes.
        x0: Vetor inicial.
        tol: Tolerância para convergência.
        max_iterations: Número máximo de iterações.

    Returns:
        x_new: Vetor solução.
        k + 1: Número de iterações realizadas.
    """
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        print(f'Iteração {k + 1}: {x_new}')  # Exibindo a iteração
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    
    raise Exception(f'Não convergiu após {max_iterations} iterações')

def eliminacao_gauss(A, b):
    """
    Método de Eliminação de Gauss para resolver sistemas lineares.
    
    Args:
        A: Matriz dos coeficientes.
        b: Vetor dos termos independentes.

    Returns:
        x: Vetor solução.
    """
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] = Ab[j] - factor * Ab[i]
    
    x = np.zeros_like(b)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return x

def is_diagonally_dominant(A):
    """
    Verifica se a matriz A é diagonalmente dominante.
    
    Args:
        A: Matriz dos coeficientes.

    Returns:
        True se A for diagonalmente dominante, caso contrário False.
    """
    for i in range(len(A)):
        row_sum = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
        if abs(A[i, i]) < row_sum:
            return False
    return True

def tornar_diagonalmente_dominante(A, b):
    """
    Tenta tornar a matriz A diagonalmente dominante.
    
    Args:
        A: Matriz dos coeficientes.
        b: Vetor dos termos independentes.

    Returns:
        A: Matriz possivelmente modificada para ser diagonalmente dominante.
        b: Vetor possivelmente permutado.
        True se a matriz foi tornada diagonalmente dominante, caso contrário False.
    """
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            if abs(A[j, i]) >= sum(abs(A[j, k]) for k in range(n) if k != i):
                A[[i, j]] = A[[j, i]]
                b[[i, j]] = b[[j, i]]
                break

    for i in range(n):
        if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
            return A, b, False
    return A, b, True

# Função de Interpolação por Lagrange

def lagrange_interpolation(x_values, y_values, x_to_evaluate):
    """
    Interpolação de Lagrange para encontrar o valor interpolado e o polinômio.
    
    Args:
        x_values: Lista de valores de x.
        y_values: Lista de valores de y.
        x_to_evaluate: Valor de x para o qual deseja-se calcular f(x).

    Returns:
        result: Valor interpolado de f(x_to_evaluate).
        polynomial: O polinômio de Lagrange.
    """
    x = symbols('x')
    n = len(x_values)
    polynomial = 0
    
    for j in range(n):
        term = y_values[j]
        for m in range(n):
            if m != j:
                term *= (x - x_values[m]) / (x_values[j] - x_values[m])
        polynomial += term

    polynomial = simplify(polynomial)
    result = polynomial.subs(x, x_to_evaluate)
    
    return result, polynomial

# Obter os inputs do usuário para interpolação

def obter_inputs_interpolacao():
    """
    Coleta os dados do usuário para a interpolação por Lagrange.
    
    Returns:
        x_values: Lista de valores de x.
        y_values: Lista de valores de y.
        x_to_evaluate: Valor de x para o qual deseja-se calcular f(x).
    """
    n = int(input("Digite o número de pontos para interpolação: "))

    print("Digite os valores de x e f(x), separados por espaço:")
    x_values = []
    y_values = []
    for i in range(n):
        x, y = map(float, input(f"Digite x{i+1} e f(x{i+1}): ").split())
        x_values.append(x)
        y_values.append(y)

    x_to_evaluate = float(input("Digite o valor de x para calcular f(x): "))
    
    return x_values, y_values, x_to_evaluate

# Obter os inputs do usuário para sistemas lineares

def obter_inputs_usuario():
    """
    Coleta os dados do usuário para resolver sistemas lineares.
    
    Returns:
        A: Matriz dos coeficientes.
        b: Vetor dos termos independentes.
        x0: Vetor inicial.
        tol: Tolerância para convergência.
        max_iterations: Número máximo de iterações.
    """
    n = int(input("Digite o número de equações (e variáveis): "))

    print("Digite os elementos da matriz A, linha por linha:")
    A = np.zeros((n, n))
    for i in range(n):
        linha = input(f"Digite os elementos da linha {i+1}, separados por espaço: ").split()
        A[i] = [float(num) for num in linha]

    print("Digite os elementos do vetor b:")
    b = np.array([float(num) for num in input("Digite os elementos de b, separados por espaço: ").split()])

    print("Digite os elementos do vetor inicial x0:")
    x0 = np.array([float(num) for num in input("Digite os elementos de x0, separados por espaço: ").split()])

    tol = float(input("Digite o valor da tolerância: "))
    max_iterations = int(input("Digite o número máximo de iterações: "))

    return A, b, x0, tol, max_iterations

# Resolver sistema linear

def resolver_sistema_linear(opcao, A, b, x0, tol, max_iterations):
    """
    Resolve o sistema linear usando o método escolhido (Jacobi ou Gauss-Seidel).
    
    Args:
        opcao: Método escolhido (1 para Jacobi, 2 para Gauss-Seidel).
        A: Matriz dos coeficientes.
        b: Vetor dos termos independentes.
        x0: Vetor inicial.
        tol: Tolerância para convergência.
        max_iterations: Número máximo de iterações.

    Returns:
        x: Vetor solução.
        iterations: Número de iterações realizadas.
    """
    if opcao == 1:
        return jacobi(A, b, x0, tol, max_iterations)
    elif opcao == 2:
        return gauss_seidel(A, b, x0, tol, max_iterations)
    else:
        raise ValueError("Opção inválida. Escolha 1 para 'jacobi' ou 2 para 'gauss-seidel'.")

# Programa principal

def main():
    while True:
        print("Escolha o tipo de cálculo que deseja realizar:")
        print("1: Resolver Sistema Linear")
        print("2: Interpolação por Lagrange")
        escolha = int(input("Digite 1 ou 2: "))

        if escolha == 1:
            print("Escolha o método para resolver o sistema linear:")
            print("1: Método de Jacobi")
            print("2: Método de Gauss-Seidel")
            opcao = int(input("Digite 1 ou 2: "))

            A, b, x0, tol, max_iterations = obter_inputs_usuario()
            A, b, sucesso = tornar_diagonalmente_dominante(A, b)

            if sucesso:
                try:
                    x, iterations = resolver_sistema_linear(opcao, A, b, x0, tol, max_iterations)
                    metodo = 'Jacobi' if opcao == 1 else 'Gauss-Seidel'
                    print(f'Solução encontrada usando o método {metodo}: {x} em {iterations} iterações')
                except Exception as e:
                    print(str(e))
            else:
                print("A matriz não é diagonalmente dominante e o método pode não convergir.")
                usar_gauss = input("Deseja resolver usando a eliminação de Gauss? (s/n): ")
                if usar_gauss.lower() == 's':
                    x = eliminacao_gauss(A, b)
                    print(f"Solução encontrada usando Eliminação de Gauss: {x}")
                else:
                    print("O programa será encerrado.")

        elif escolha == 2:
            x_values, y_values, x_to_evaluate = obter_inputs_interpolacao()
            result, polynomial = lagrange_interpolation(x_values, y_values, x_to_evaluate)
            print(f"O valor interpolado de f({x_to_evaluate}) é {result}")
            print(f"O polinômio interpolador de Lagrange é: {polynomial}")

        else:
            print("Opção inválida. O programa será encerrado.")

        continuar = input("Deseja realizar outro cálculo? (s/n): ").lower()
        if continuar != 's':
            print("Encerrando o programa.")
            break

if __name__ == "__main__":
    main()