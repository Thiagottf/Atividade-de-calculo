import numpy as np
from sympy import symbols, simplify

# Funções para resolver sistemas lineares

def jacobi(A, b, x0, tol, max_iterations, decimal_places):
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum = 0
            for j in range(n):
                if i != j:
                    sum += A[i, j] * x[j]
            print(f'Somatório na equação {i+1} (iteração {k+1}): {sum}')  # Exibe o somatório parcial para cada equação
            x_new[i] = (b[i] - sum) / A[i, i]
        
        x_new = np.round(x_new, decimal_places)  # Arredondar os resultados
        print(f'Iteração {k + 1}: {x_new}\n')  # Exibindo a iteração
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    
    raise Exception(f'Não convergiu após {max_iterations} iterações')

def gauss_seidel(A, b, x0, tol, max_iterations, decimal_places):
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            sum_total = sum1 + sum2
            print(f'Somatório na equação {i+1} (iteração {k+1}): {sum_total}')  # Exibe o somatório parcial para cada equação
            x_new[i] = (b[i] - sum_total) / A[i, i]
        
        x_new = np.round(x_new, decimal_places)  # Arredondar os resultados
        print(f'Iteração {k + 1}: {x_new}\n')  # Exibindo a iteração
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    
    raise Exception(f'Não convergiu após {max_iterations} iterações')

def eliminacao_gauss(A, b, decimal_places):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] = Ab[j] - factor * Ab[i]
    
    x = np.zeros_like(b)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return np.round(x, decimal_places)  # Arredondar os resultados

def is_diagonally_dominant(A):
    for i in range(len(A)):
        row_sum = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
        if abs(A[i, i]) < row_sum:
            return False
    return True

def tornar_diagonalmente_dominante(A, b):
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

def lagrange_interpolation(x_values, y_values, x_to_evaluate, decimal_places):
    x = symbols('x')
    n = len(x_values)
    polynomial = 0

    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        polynomial += term

    polynomial_simplified = simplify(polynomial)
    result = polynomial_simplified.subs(x, x_to_evaluate)

    # Arredondar o resultado para o número de casas decimais escolhido
    result = round(result, decimal_places)

    return result, polynomial_simplified

# Obter os inputs do usuário para interpolação

def obter_inputs_interpolacao():
    n = int(input("Digite o número de pontos para interpolação: "))

    print("Digite os valores de x e f(x), separados por espaço:")
    x_values = []
    y_values = []
    for i in range(n):
        x, y = map(float, input(f"Digite x{i+1} e f(x{i+1}): ").split())
        x_values.append(x)
        y_values.append(y)

    x_to_evaluate = float(input("Digite o valor de x para calcular f(x): "))
    decimal_places = int(input("Digite o número de casas decimais para o resultado: "))
    
    return x_values, y_values, x_to_evaluate, decimal_places

# Obter os inputs do usuário para sistemas lineares

def obter_inputs_usuario():
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
    decimal_places = int(input("Digite o número de casas decimais para os resultados: "))

    return A, b, x0, tol, max_iterations, decimal_places

# Resolver sistema linear

def resolver_sistema_linear(opcao, A, b, x0, tol, max_iterations, decimal_places):
    if opcao == 1:
        return jacobi(A, b, x0, tol, max_iterations, decimal_places)
    elif opcao == 2:
        return gauss_seidel(A, b, x0, tol, max_iterations, decimal_places)
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

            A, b, x0, tol, max_iterations, decimal_places = obter_inputs_usuario()
            A, b, sucesso = tornar_diagonalmente_dominante(A, b)

            if sucesso:
                try:
                    x, iterations = resolver_sistema_linear(opcao, A, b, x0, tol, max_iterations, decimal_places)
                    metodo = 'Jacobi' if opcao == 1 else 'Gauss-Seidel'
                    print(f'Solução encontrada usando o método {metodo}: {x} em {iterations} iterações')
                except Exception as e:
                    print(str(e))
            else:
                print("A matriz não é diagonalmente dominante e o método pode não convergir.")
                usar_gauss = input("Deseja resolver usando a eliminação de Gauss? (s/n): ")
                if usar_gauss.lower() == 's':
                    x = eliminacao_gauss(A, b, decimal_places)
                    print(f"Solução encontrada usando Eliminação de Gauss: {x}")
                else:
                    print("O programa será encerrado.")

        elif escolha == 2:
            x_values, y_values, x_to_evaluate, decimal_places = obter_inputs_interpolacao()
            result, polynomial = lagrange_interpolation(x_values, y_values, x_to_evaluate, decimal_places)
            print(f'Polinômio de Lagrange: {polynomial}')
            print(f'Valor interpolado para x = {x_to_evaluate}: f(x) = {result}')
        else:
            print("Opção inválida. Encerrando o programa.")
            break

if __name__ == "__main__":
    main()
