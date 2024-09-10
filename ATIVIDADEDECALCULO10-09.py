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
            x_new[i] = (b[i] - sum) / A[i, i]
            # Printar as somatórias
            print(f'Somatório da equação {i+1} na iteração {k+1}: {sum}')

        print(f'Iteração {k + 1}: {x_new}')  # Exibindo a iteração

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            # Arredondar o resultado final
            return np.round(x_new, decimal_places), k + 1
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
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
            # Printar as somatórias
            print(f'Somatório antes do i {i+1} na iteração {k+1}: sum1 = {sum1}, sum2 = {sum2}')

        print(f'Iteração {k + 1}: {x_new}')  # Exibindo a iteração

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            # Arredondar o resultado final
            return np.round(x_new, decimal_places), k + 1
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

    # Arredondar o resultado final
    return np.round(x, decimal_places)

# Função de Interpolação por Lagrange
def lagrange_interpolation(x_values, y_values, x_to_evaluate, decimal_places):
    x = symbols('x')
    n = len(x_values)
    polynomial = 0

    for i in range(n):
        term = y_values[i]
        print(f"\nCalculando o termo {i + 1}:")
        for j in range(n):
            if i != j:
                print(f"Multiplicando por (x - {x_values[j]}) / ({x_values[i]} - {x_values[j]})")
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        print(f"Termo {i + 1} antes de simplificação: {term}")
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
    decimal_places = int(input("Digite o número de casas decimais para o resultado: "))

    return A, b, x0, tol, max_iterations, decimal_places

# Função para verificar e tornar a matriz diagonalmente dominante
def tornar_diagonalmente_dominante(A, b):
    n = len(A)
    sucesso = False
    for i in range(n):
        soma = sum(abs(A[i][j]) for j in range(n) if j != i)
        if abs(A[i][i]) >= soma:
            sucesso = True
        else:
            sucesso = False
            break

    return A, b, sucesso

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
            print(f'\nPolinômio de Lagrange: {polynomial}')
            print(f'Valor interpolado para x = {x_to_evaluate}: f(x) = {result}')

        else:
            print("Opção inválida. O programa será encerrado.")

        continuar = input("Deseja realizar outro cálculo? (s/n): ").lower()
        if continuar != 's':
            print("Encerrando o programa.")
            break

if __name__ == "__main__":
    main()
