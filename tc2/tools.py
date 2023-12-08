import numpy as np
from scipy import optimize


def external_penalty(current_point, ineq_constraints, eq_constraints, u, fx):
    ineq = [
        u * (max(0, g(current_point)) ** 2)
        for g in ineq_constraints
    ]
    eq = [
        u * (h(current_point) ** 2)
        for h in eq_constraints
    ]
    f = fx(current_point)
    return f + sum(ineq) + sum(eq)

def solve(
        fx: callable,
        ineq_constraints: list[callable],
        eq_constraints: list[callable],
        initial_point,
        method='BFGS',
        max_iterations: float = np.inf,
        precisao = 1e-2
):
    # Parâmetros
    u = 1.  # Valor inicial de u
    alpha = 1.5  # Aceleração do valor de u
    xlast = np.inf * np.ones(len(initial_point))  # Último valor de u
    iteracoes = 1  # Contador de iterações

    while iteracoes < max_iterations:
        # Determina o ponto de ótimo através de um método de otimização irrestrita
        solution = optimize.minimize(
            external_penalty,
            initial_point,
            args=(ineq_constraints, eq_constraints, u, fx),
            method=method)
        xopt = solution.x
        fopt = solution.fun

        # Se o percentual de diferença entre os dois últimos ótimos forem muito pequenos, pare
        if np.linalg.norm((xopt - xlast) / xopt) < precisao:
            break

        # Senão, aumente u
        else:
            xlast = xopt
            u = alpha * u
            iteracoes += 1

    # Exibe resultado
    print('RESULTADO')
    print('x-ótimo: ' + str(xopt))
    print('f(x-ótimo): %.3f' % fopt)
    print('Valor final de u: %.1f' % u)
    print('Número de iterações: %d' % iteracoes)


def base_test():
    def fx(x):
        x1, x2 = x[0], x[1]
        return (x1-3)**2 + 2*(x2-3)**2

    def gx(x):
        x1, x2 = x[0], x[1]
        return 3*x1 + 2*x2 - 12

    def hx(x):
        x1, x2 = x[0], x[1]
        return x1 + x2 - 5

    solve(
        fx=fx,
        ineq_constraints=[gx],
        eq_constraints=[hx],
        initial_point=np.array([0, 0])
    )


def q1_minimizao_material_caixa():
    """
    Problema de otimização restrita que busca minimizar o consumo de material para construção de uma caixa. Esse consumo
    é definido pela área superficial da caixa.
    :return:
    """
    def fx(x):
        a, b, c = x
        return 1.5 * (2 * a * b + 2 * a * c + 2 * b * c)

    def volume(x):
        return np.prod(x) - 0.032

    def perimetro_base(x):
        a, b, c = x
        return 2 * (a + b) - 1.5

    def largura(x):
        a, b, c = x
        return b - 3 * a

    def altura(x):
        a, b, c = x
        return c - (2 / 3) * b

    def comprimento_maximo(x):
        return x[0] - 0.5

    def g5(x):
        return -x[0]

    def g6(x):
        return -x[1]

    def largura_maxima(x):
        return x[1] - 0.5

    def g8(x):
        return -x[2]

    solve(
        fx=fx,
        eq_constraints=[volume],
        ineq_constraints=[
            perimetro_base,
            largura,
            altura,
            comprimento_maximo,
            largura_maxima,
            g5, g6, g8
        ],
        initial_point=[1, 1, 1],
        method='BFGS',
        max_iterations=100,
        precisao=1e-3
    )


def q2():
    pass


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    q1_minimizao_material_caixa()
