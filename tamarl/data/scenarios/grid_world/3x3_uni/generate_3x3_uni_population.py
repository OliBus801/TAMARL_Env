import argparse
import os


def generate_population(n_agents: int, output_file: str):
    # Les 6 chemins possibles de 0_0 à 2_2 en se déplaçant vers la droite et le haut
    routes = [
        ["0_0-0_1", "0_1-0_2", "0_2-1_2", "1_2-2_2"],  # Haut, Haut, Droite, Droite
        ["0_0-0_1", "0_1-1_1", "1_1-1_2", "1_2-2_2"],  # Haut, Droite, Haut, Droite
        ["0_0-0_1", "0_1-1_1", "1_1-2_1", "2_1-2_2"],  # Haut, Droite, Droite, Haut
        ["0_0-1_0", "1_0-1_1", "1_1-1_2", "1_2-2_2"],  # Droite, Haut, Haut, Droite
        ["0_0-1_0", "1_0-1_1", "1_1-2_1", "2_1-2_2"],  # Droite, Haut, Droite, Haut
        ["0_0-1_0", "1_0-2_0", "2_0-2_1", "2_1-2_2"],  # Droite, Droite, Haut, Haut
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(
            '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">\n'
        )
        f.write("<population>\n")

        for i in range(n_agents - 1, -1, -1):
            pid = f"agent_{i:03d}"
            # Distribution équitable en itérant successivement à travers les 6 chemins
            route = routes[i % 6]
            start_link = route[0]
            end_link = route[-1]
            route_str = " ".join(route)

            f.write(f'    <person id="{pid}">\n')
            f.write('        <plan selected="yes">\n')
            f.write(
                f'            <act type="h" link="{start_link}" x="0.0" y="0.0" end_time="00:00:00" />\n'
            )
            f.write('            <leg mode="car">\n')
            f.write(
                f'                <route type="links" start_link="{start_link}" end_link="{end_link}">{route_str}</route>\n'
            )
            f.write("            </leg>\n")
            # 1000.0, 1000.0 car c'est la coordonnée du noeud 2_2 dans le réseau 3x3_network
            f.write(f'            <act type="w" link="{end_link}" x="1000.0" y="1000.0" />\n')
            f.write("        </plan>\n")
            f.write("    </person>\n")

        f.write("</population>\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MATSim Population for 3x3_uni")
    parser.add_argument("N", type=int, help="Nombre d'agents à générer")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = f"{args.N}_population.xml"
    output_path = os.path.join(script_dir, output_filename)

    generate_population(args.N, output_path)
    print(
        f"[{output_filename}] généré avec succès dans le dossier {script_dir} pour {args.N} agents."
    )
