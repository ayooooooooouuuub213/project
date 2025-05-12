import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import time

def knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon):
    start = time.time()
    decision = cp.Variable(len(poids_object), boolean=True)
    contrainte_poid = poids_object @ decision <= poids_maximum
    Z1 = z1_vect @ decision
    p1 = cp.Problem(cp.Maximize(Z1), [contrainte_poid])
    p1.solve(solver=cp.GLPK_MI)

    Z2 = z2_vect @ decision
    p2 = cp.Problem(cp.Maximize(Z2), [contrainte_poid, Z1 == Z1.value])
    p2.solve(solver=cp.GLPK_MI)

    solutions_pareto = [decision.value]
    j = 1
    valeurs_solution_Z1 = [Z1.value]
    valeurs_solution_Z2 = [Z2.value]
    solutions_Z = [(Z1.value, Z2.value)]

    while True:
        p_epsilon = cp.Problem(cp.Maximize(Z1), [
            contrainte_poid,
            Z2 >= valeurs_solution_Z2[j - 1] + epsilon
        ])
        p_epsilon.solve(solver=cp.GLPK_MI)

        if Z1.value is None or Z2.value is None:
            break
        else:
            solutions_pareto.append(decision.value)
            valeurs_solution_Z1.append(Z1.value)
            valeurs_solution_Z2.append(Z2.value)
            solutions_Z.append((Z1.value, Z2.value))
            j += 1
    end = time.time()

    st.write("### L'ensemble des solutions efficaces:")
    indexes = [f'X({i+1})' for i in range(len(solutions_pareto))]
    df = pd.DataFrame({
        '': indexes,
        'Solution': solutions_pareto,
        'Z1': valeurs_solution_Z1,
        'Z2': valeurs_solution_Z2
    })
    with st.expander("Solutions"):
        st.table(df.set_index(df.columns[0]))

    st.write("### Front de Pareto :")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(valeurs_solution_Z1, valeurs_solution_Z2, marker='o', linestyle='-')
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_title('Front de Pareto')
    st.pyplot(fig)

    st.write(f'### Temps de calcul : {round((end - start), 4)} secondes')


def main():
    selected = option_menu(
        menu_title="",
        options=["Manual", "Random"],
        default_index=0,
        orientation='horizontal'
    )

    st.markdown("**Développé par : HAMDANE Ayoub et HAMMACHE Ghiles**")
    st.title("Bi-objective Knapsack Problem Solver (Méthode epsilon-contrainte)")

    poids_maximum = st.number_input("Poids maximum du sac :", min_value=0.0)
    n = st.number_input("Nombre d'objets :", step=1, min_value=1)
    epsilon = st.number_input("Valeur d'epsilon :", min_value=0.0)

    if selected == "Manual":
        poids_object = np.array([])
        z1_vect = np.array([])
        z2_vect = np.array([])

        st.write("### Définir les objets :")
        with st.expander("Objets"):
            for i in range(int(n)):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f'Objet {i + 1}')
                with col2:
                    poid = st.number_input(f"Poids :", key=f"poids_{i}", min_value=0.0)
                with col3:
                    valeur1 = st.number_input(f"Valeur Z1 :", key=f"z1_{i}", min_value=0.0)
                with col4:
                    valeur2 = st.number_input(f"Valeur Z2 :", key=f"z2_{i}", min_value=0.0)

                poids_object = np.append(poids_object, poid)
                z1_vect = np.append(z1_vect, valeur1)
                z2_vect = np.append(z2_vect, valeur2)

        if st.button("Résoudre"):
            knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon)

    elif selected == "Random":
        if st.button("Générer aléatoirement"):
            poids_object = np.random.uniform(1, poids_maximum, int(n))
            z1_vect = np.random.uniform(1, 100, int(n))
            z2_vect = np.random.uniform(1, 100, int(n))

            st.write("### Objets générés aléatoirement :")
            df = pd.DataFrame({
                'Poids': poids_object,
                'Z1': z1_vect,
                'Z2': z2_vect
            })
            st.table(df)

            knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon)


if __name__ == "__main__":
    main()
