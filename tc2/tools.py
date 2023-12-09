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


def internal_penalty(current_point, ineq_constraints, eq_constraints, u, fx):
    ineq = [
        1/g(current_point)
        for g in ineq_constraints
    ]

    f = fx(current_point)
    return f - u*sum(ineq)


def augmented_lagrangian(
        current_point,
        ineq_constraints,
        eq_constraints,
        u,
        fx,
        mus=[],
        lambs=[],
):
    ineq = [
        max(g(current_point), -mu/u) ** 2
        for g, mu in zip(ineq_constraints, mus)
    ]
    eq = [
        (h(current_point) + lamb/u) ** 2
        for h, lamb in zip(eq_constraints, lambs)
    ]
    f = fx(current_point)
    return f + u/2 * (sum(ineq) + sum(eq))


def solve_penalties(
        fx: callable,
        ineq_constraints: list[callable],
        eq_constraints: list[callable],
        initial_point,
        method='BFGS',
        penalty=external_penalty,
        max_iterations: float = np.inf,
        precisao = 1e-2,
        alpha=1.5  # Aceleração do valor de u
):
    # Parâmetros
    u = 1.  # Valor inicial de u
    xlast = np.inf * np.ones(len(initial_point))  # Último valor de u
    iteracoes = 1  # Contador de iterações

    while iteracoes < max_iterations:
        # Determina o ponto de ótimo através de um método de otimização irrestrita
        solution = optimize.minimize(
            penalty,
            initial_point,
            args=(ineq_constraints, eq_constraints, u, fx),
            method=method
        )
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


def solve_lagrangian(
        fx: callable,
        ineq_constraints: list[callable],
        eq_constraints: list[callable],
        initial_point,
        method='BFGS',
        max_iterations: float = np.inf,
        precisao = 1e-2,
        alpha=1.5  # Aceleração do valor de u
):
    # Parâmetros
    u = 1.  # Valor inicial de u
    xlast = np.inf * np.ones(len(initial_point))  # Último valor de u
    iteracoes = 1  # Contador de iterações
    lambs = [0 for _ in eq_constraints]
    mus = [0 for _ in ineq_constraints]
    xopt = initial_point
    while iteracoes < max_iterations:
        # Determina o ponto de ótimo através de um método de otimização irrestrita
        solution = optimize.minimize(
            augmented_lagrangian,
            xopt,
            args=(ineq_constraints, eq_constraints, u, fx, mus, lambs),
            method=method
        )
        xopt = solution.x
        fopt = solution.fun

        # Exib
        # e resultado
        print('Iteração %d' % iteracoes, end=' - ')
        print('x-ótimo: ' + str(xopt), end=', ')
        print('f(x-ótimo): ' + str(fopt), end=', ')
        for i, lamb in enumerate(lambs):
            print(f'lambda {i} = {lamb}', end=', ')
        # for i, mu in enumerate(mus):
        print(f'mu`s = {mus}', end=', ')
        print('u = %.2f' % u)

        # Se o percentual de diferença entre os dois últimos ótimos forem muito pequenos, pare
        if np.linalg.norm((xopt - xlast) / xopt) < precisao:
            break

        # Senão, aumente u
        else:
            xlast = xopt
            for i, mu in enumerate(mus):
                mus[i] = mu + u*max(ineq_constraints[i](xopt), -mu/u)
            for i, lamb in enumerate(lambs):
                lambs[i] = lamb + u*eq_constraints[i](xopt)
            u = alpha * u
            iteracoes += 1




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

    solve_penalties(
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

    solve_penalties(
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


def q2_minimize_peso_trelica():
    def stress_trus_1(x):
        x1, x2 = x
        p = 20
        r2 = np.sqrt(2)
        num = x2 + x1*r2
        den = (x1**2)*r2 + 2*x1*x2
        return p * (num/den) - 20

    def stress_trus_2(x):
        x1, x2 = x
        p = 20
        r2 = np.sqrt(2)
        num = 1
        den = x1 + x2*r2
        return p * (num/den) - 20

    def stress_trus_3(x):
        x1, x2 = x
        p = 20
        r2 = np.sqrt(2)
        num = x2
        den = (x1**2)*r2 + 2*x1*x2
        return -p * (num/den) + 15

    def trus_weight(x):
        return 2*np.sqrt(2)*x[0] + x[1]

    solve_lagrangian(
        fx=trus_weight,
        ineq_constraints=[
            stress_trus_1,
            stress_trus_2,
            stress_trus_3
        ],
        eq_constraints=[],
        initial_point=[2, 2],
        precisao=1e-4
    )


def q3_rosenbrock():
    def rosenbrock(x):
        a, b = x
        return (1 - a) ** 2 + 100 * (b - a ** 2) ** 2

    def g1(vars):
        a, b = vars
        return (a - 1) ** 3 + 1

    def g2(x):
        a, b = x
        return -a + 2

    solve_penalties(
        fx=rosenbrock,
        eq_constraints=[],
        ineq_constraints=[
            g1, g2
        ],
        initial_point=[1, 1.5],
        method='BFGS',
        max_iterations=100,
        precisao=1e-4,
        penalty=internal_penalty,
        alpha=0.5
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    print("Questão 1: Minimização de área de caixa")
    q1_minimizao_material_caixa()
    print("="*50)
    print("\nQuestão 2: Minimização de peso de Treliça")
    q2_minimize_peso_trelica()
    print("="*50)
    print("\nQuestão 3: Rosenbrock")
    q3_rosenbrock()
