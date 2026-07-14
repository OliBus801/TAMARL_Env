#!/usr/bin/env python3
import argparse
import os
import xml.etree.ElementTree as ET


def scale_matsim_network(input_path, scale_factor, output_path, eff_cell_size=7.5):
    """
    Reads a MATSim network XML file, scales its flow capacity and storage capacity
    for each link, and writes the scaled network to a new XML file.

    If 'storageCapacity' is not present in a link, it is calculated as:
        (length * permlanes) / eff_cell_size
    before applying the scale factor.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading network from: {input_path}")
    print(f"Scale factor: {scale_factor}")
    print(f"Effective cell size: {eff_cell_size} meters")

    # Attempt to read the original XML declaration and DOCTYPE
    xml_declaration = None
    doctype_line = None
    try:
        with open(input_path, encoding="utf-8") as f:
            for _ in range(15):  # read first 15 lines to find DOCTYPE or XML decl
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith("<?xml"):
                    xml_declaration = stripped
                elif stripped.startswith("<!DOCTYPE"):
                    doctype_line = stripped
                    break
    except Exception as e:
        print(f"Warning: Could not read DOCTYPE/XML declaration header: {e}")

    # Use insert_comments=True if available in python 3.8+
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(input_path, parser=parser)
    except Exception as e:
        print(
            f"Warning: Failed to parse with comments preserved: {e}. Falling back to standard parser."
        )
        tree = ET.parse(input_path)

    root = tree.getroot()

    links_count = 0
    scaled_count = 0

    # Find the links container and its link children
    for link in root.findall(".//link"):
        links_count += 1

        # 1. Scale flow capacity
        capacity_str = link.get("capacity")
        if capacity_str is not None:
            try:
                capacity = float(capacity_str)
                new_capacity = capacity * scale_factor
                link.set("capacity", f"{new_capacity:.4f}")
            except ValueError:
                print(
                    f"Warning: Could not parse capacity '{capacity_str}' for link id {link.get('id')}"
                )

        # 2. Scale storage capacity
        storage_cap_str = link.get("storageCapacity")
        if storage_cap_str is not None:
            try:
                storage_cap = float(storage_cap_str)
                new_storage_cap = storage_cap * scale_factor
                link.set("storageCapacity", f"{new_storage_cap:.4f}")
                scaled_count += 1
            except ValueError:
                print(
                    f"Warning: Could not parse storageCapacity '{storage_cap_str}' for link id {link.get('id')}"
                )
        else:
            # Calculate default storage capacity if not present
            length_str = link.get("length")
            permlanes_str = link.get("permlanes")
            if length_str is not None:
                try:
                    length = float(length_str)
                    lanes = float(permlanes_str) if permlanes_str is not None else 1.0

                    # Estimate original storage capacity
                    storage_cap = (length * lanes) / eff_cell_size
                    new_storage_cap = storage_cap * scale_factor
                    link.set("storageCapacity", f"{new_storage_cap:.4f}")
                    scaled_count += 1
                except ValueError:
                    print(f"Warning: Could not parse length/permlanes for link id {link.get('id')}")
            else:
                print(f"Warning: Link id {link.get('id')} has neither storageCapacity nor length.")

    print(
        f"Processed {links_count} links. Scaled capacity on all, set/scaled storageCapacity on {scaled_count} links."
    )

    # Ensure directory exists for output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write back the scaled network XML
    with open(output_path, "wb") as f:
        # Prepend headers
        if xml_declaration:
            f.write(xml_declaration.encode("utf-8") + b"\n")
        else:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')

        if doctype_line:
            f.write(doctype_line.encode("utf-8") + b"\n")
        else:
            f.write(b'<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">\n')

        # Write modified XML
        tree.write(f, encoding="utf-8", xml_declaration=False)

    print(f"Scaled network written successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scale flow capacity and storage capacity of a MATSim network XML file."
    )
    parser.add_argument("input_file", help="Path to the input MATSim network XML file.")
    parser.add_argument(
        "scale_factor", type=float, help="Scale factor (e.g. 1.1 for 110%%, 0.5 for 50%%)."
    )
    parser.add_argument("output_file", help="Path to save the scaled MATSim network XML file.")
    parser.add_argument(
        "--eff_cell_size",
        type=float,
        default=7.5,
        help="Effective cell size in meters per vehicle for storageCapacity calculation (default: 7.5).",
    )

    args = parser.parse_args()

    scale_matsim_network(
        input_path=args.input_file,
        scale_factor=args.scale_factor,
        output_path=args.output_file,
        eff_cell_size=args.eff_cell_size,
    )
