import argparse
import copy
import os
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(
        description="Scale Sioux Falls population XML by an integer factor."
    )
    parser.add_argument(
        "--scale", type=int, default=2, help="Scale factor (e.g. 2 to double, 3 to triple)"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Input population XML file name or path"
    )
    args = parser.parse_args()

    scale = args.scale
    if scale < 1:
        print("Error: Scale factor must be >= 1.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve input path
    if args.input:
        if os.path.isabs(args.input):
            input_path = args.input
        else:
            input_path = os.path.join(script_dir, args.input)
    else:
        # Default: try routed first, then standard
        routed_path = os.path.join(script_dir, "Siouxfalls_route_population.xml")
        std_path = os.path.join(script_dir, "Siouxfalls_population.xml")
        if os.path.exists(routed_path):
            input_path = routed_path
        else:
            input_path = std_path

    pct = scale * 100
    output_path = os.path.join(script_dir, f"Siouxfalls_{pct}pct_population.xml")

    print(f"Reading from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Parse XML
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Find all person elements
    persons = root.findall("person")
    print(f"Found {len(persons)} persons. Scaling population by {scale}x...")

    # Duplicate population S - 1 times
    for person in persons:
        orig_id = person.attrib.get("id", "")
        for c in range(1, scale):
            cloned_person = copy.deepcopy(person)
            # Adapt the ID to prevent collisions
            if scale == 2:
                new_id = f"{orig_id}_cloned"
            else:
                new_id = f"{orig_id}_cloned_{c}"
            cloned_person.attrib["id"] = new_id
            root.append(cloned_person)

    print(f"Writing to: {output_path}")
    # Write output with the proper MATSim DOCTYPE header
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    header = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v5.dtd">\n'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(xml_str)

    print(f"Done! Population successfully scaled to {pct}%.")


if __name__ == "__main__":
    main()
