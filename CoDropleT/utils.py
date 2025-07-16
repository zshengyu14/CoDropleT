# utils.py

def update_pdb_b_factors(pdb_file, output_file, b_factors):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    residue_index = -1
    last_residue = None

    with open(output_file, 'w') as file:
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                chain = line[21]
                residue_id = int(line[22:26].strip())
                residue_name = line[17:20].strip()
                res_tuple = (chain, residue_id, residue_name)
                if res_tuple != last_residue:
                    last_residue = res_tuple
                    residue_index += 1
                b_factor_str = f"{b_factors[residue_index]:6.2f}"
                line = f"{line[:60]}{b_factor_str}{line[66:]}"
            file.write(line)
    assert residue_index == len(b_factors) - 1
