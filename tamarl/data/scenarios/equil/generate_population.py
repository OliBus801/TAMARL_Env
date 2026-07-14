import sys


def generer_population(n_agents, fichier_sortie):
    entete = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">
<population>"""

    pied_page = "\n</population>"

    gabarit_personne = """
    <person id="agent_{id:03d}">
        <plan selected="yes">
            <act type="h" link="1-2" end_time="00:00:00" />
            <leg mode="car">
                <route type="links" start_link="1-2" end_link="12-13">1-2 2-7 7-12 12-13</route>
            </leg>
            <act type="w" link="12-13" duration="00:30:00" />
            <leg mode="car">
                <route type="links" start_link="12-13" end_link="15-1">12-13 13-14 14-15 15-1</route>
            </leg>
            <act type="h" link="1-2" />
        </plan>
    </person>"""

    with open(fichier_sortie, "w", encoding="utf-8") as f:
        f.write(entete)

        # Boucle en ordre décroissant (de N-1 jusqu'à 0)
        for i in range(n_agents - 1, -1, -1):
            f.write(gabarit_personne.format(id=i))

        f.write(pied_page)

    print(f"Génération terminée : {n_agents} agents dans {fichier_sortie}.")


if __name__ == "__main__":
    # Utilisation : python script.py 1000 population.xml
    if len(sys.argv) == 3:
        try:
            n = int(sys.argv[1])
            generer_population(n, sys.argv[2])
        except ValueError:
            print("Erreur : Le premier argument doit être un entier.")
    else:
        print("Usage: python script.py <nombre_agents> <fichier_sortie.xml>")
