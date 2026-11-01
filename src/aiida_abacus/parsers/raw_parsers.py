import re
from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np

logger = getLogger(__name__)


class BaseRawParser:
    def __init__(self, fhandle):
        """A parser for the ABACUS output file."""
        if not hasattr(fhandle, "read"):
            self.content = Path(fhandle).read_text()
        else:
            self.content = fhandle.read()
        self.lines = self.content.split("\n")


class AbacusRawParser(BaseRawParser):
    """
    A parser to process abacus output files (running_xxx.log)
    """

    def __init__(self, fhandle):
        """A parser for the ABACUS output file."""
        super().__init__(fhandle)
        self.results = {}
        self.is_parsed = False

    def parse_blocks(self) -> None:
        """
        Parse blocks from output file.

        Supports both the v3.10 layout::

            ---------
             TOTAL-FORCE (eV/Angstrom)
            ------
            Si1 0.0 0.0 0.0
            ------

        and the v3.11 layout::

             #TOTAL-FORCE (eV/Angstrom)#
            ------------------------------------
                Atoms     Force_x  Force_y  Force_z
            ------------------------------------
            Si1 0.0 0.0 0.0
            ------------------------------------
        """
        # v3.10
        p_old = re.compile(
            r"-+\n TOTAL-(FORCE|STRESS) \(([a-zA-Z/]+)\) *\n\s*-+\n(.*?)\n\s*-+",
            flags=re.DOTALL,
        )
        # v3.11: optional column-header line between the two separator lines
        p_new = re.compile(
            r"#\s*TOTAL-(FORCE|STRESS) \(([a-zA-Z/]+)\)\s*#\s*\n\s*-+\n(?:[^\n]*\n)?\s*-+\n(.*?)\n\s*-+",
            flags=re.DOTALL,
        )

        all_blocks = []
        seen_spans = []
        for pattern in (p_old, p_new):
            for match in pattern.finditer(self.content):
                if any(
                    match.start() < end and match.end() > start
                    for start, end in seen_spans
                ):
                    continue  # overlap: same block already captured by the other pattern
                seen_spans.append((match.start(), match.end()))
                all_blocks.append((match.group(1), match.group(2), match.group(3)))

        atom_label = re.compile(r"^[A-Za-z]{1,3}\d*$")
        all_forces = []
        all_stress = []
        for block_type, block_unit, block_content in all_blocks:
            lines = []
            for line in block_content.split("\n"):
                tokens = line.split()
                if not tokens:
                    continue
                try:
                    if block_type == "FORCE":
                        # atom label then Fx Fy Fz
                        if len(tokens) < 4 or not atom_label.match(tokens[0]):
                            continue
                        values = [float(token) for token in tokens[1:]]
                    else:
                        if len(tokens) < 3:
                            continue
                        values = [float(token) for token in tokens]
                except ValueError:
                    continue
                lines.append(values)
            if lines:
                if block_type == "FORCE":
                    all_forces.append(lines)
                    self.results["force_unit"] = block_unit
                else:
                    all_stress.append(lines)
                    self.results["stress_unit"] = block_unit

        self.results["all_forces"] = all_forces
        self.results["all_stress"] = all_stress
        self.results["final_forces"] = all_forces[-1] if all_forces else None
        self.results["final_stress"] = all_stress[-1] if all_stress else None

    def parse(self) -> dict:
        """
        Parse ABACUS output file.

        :returns: parsed results as a dictionary
        """
        self.parse_blocks()
        # Parse the lines one-by-one for general information of the calculation
        for line in self.lines:
            if "TOTAL-stress" in line:
                self.results["stress"] = line.strip().split()[1]
                self.results["stress_unit"] = line.strip().split()[2]
                if "all_stress" not in self.results:
                    self.results["all_stress"] = []
                self.results["all_stress"].append(self.results["stress"])
                continue
            elif "!FINAL_ETOT_IS" in line:
                self.results["total_energy"] = line.strip().split()[1]
            elif "NBANDS =" in line:
                self.results["number_of_bands"] = int(line.strip().split()[-1])
            elif "EFERMI" in line:
                self.results["fermi_level"] = float(line.strip().split()[-2])

        self.is_parsed = True
        return self.results

    def parse_kpoints(self):
        """
        Parse the kpoints involved in the calculation

        :return: A tuple of kpoints in direct and cartesian coordinates
        """

        kdirect = BlockParser(
            self.lines,
            re.compile(r"^K-POINTS (DIRECT) COORDINATES"),
            offset=2,
            types=[int, float, float, float, float],
        ).parse()
        kcart = BlockParser(
            self.lines,
            re.compile(r"^K-POINTS (CARTESIAN) COORDINATES"),
            offset=2,
            types=[int, float, float, float, float],
        ).parse()
        if len(kdirect) == 0:
            raise ValueError("No kpoints data found")
        if len(kdirect) > 2:
            raise ValueError("Multiple sets of kpoints data found")
        # Take the last set of kpoint reported
        # Return an array made of kpoint coordinates and weight, remove the kpoint index
        return np.array(kdirect[-1][1])[:, 1:], np.array(kcart[-1][1])[:, 1:]

    def parse_eigenvalues(self):
        """
        Parse the eigenvalues
        :return: A tuple of eigenvalues and occupations and k-points (in cartesian coordinates)
        """

        nspins = int(re.search(r"NSPIN == (\d)", self.content).group(1))
        nkthis_procs = int(
            re.search(r"k-point number in this process = (\d+)", self.content).group(1)
        )
        parser = BlockParser(
            self.lines,
            re.compile(
                r"^ (\d+)/(\d+) kpoint \(Cartesian\) *= *([-0-9.]+) ([-0-9.]+) ([-0-9.]+)"
            ),
            offset=1,
            types=[int, float, float],
        )
        blocks = parser.parse()
        eigenvalues = {}
        occupations = {}
        ntot = len(blocks)
        nkpts = ntot // nspins
        # NOTE: Abacus only report the kpoint on the head MPI process!
        # TODO: Raise a PR to the developers to include all kpoints in the log file.
        if nkpts != nkthis_procs:
            logger.wanning("The number of kpoint is (), but only () on this proc")
        assert ntot % nspins == 0
        kpt_cart = np.zeros((nkpts, 3))
        # Process all blocks
        for i, (key, block) in enumerate(blocks):
            ikpt = int(key[0])
            # Sanity check
            if i == 0:
                nkpt_tot = int(key[1])
                assert nkpt_tot == nkpts, (
                    "Mismatch in kpont number possible unsupported spin type"
                )
            kpt_cart[ikpt - 1, 0] = float(key[2])
            kpt_cart[ikpt - 1, 1] = float(key[3])
            kpt_cart[ikpt - 1, 2] = float(key[4])
            # Check which spin we are with
            ispin = i // nkpts
            if ispin not in eigenvalues:
                eigenvalues[ispin] = {}
                occupations[ispin] = {}
            occ = [entry[2] for entry in block]
            energy = [entry[1] for entry in block]
            eigenvalues[ispin][ikpt] = np.array(energy)
            occupations[ispin][ikpt] = np.array(occ)
        # Construct overall block
        nkpts = len(eigenvalues[0])
        nspins = len(eigenvalues)
        assert max(eigenvalues[0].keys()) == nkpts
        eigen_arrays = []
        occ_arrays = []
        for spin in range(nspins):
            eigen_arrays.append(
                np.stack([eigenvalues[spin][i] for i in range(1, nkpts + 1)], axis=0)
            )
            occ_arrays.append(
                np.stack([occupations[spin][i] for i in range(1, nkpts + 1)], axis=0)
            )
        return np.stack(eigen_arrays, axis=0), np.stack(occ_arrays, axis=0), kpt_cart


class BlockParser:
    """Parser to extract blocks of data"""

    DEFAULT_END_CHAR = ["------", "++++++"]

    def __init__(
        self, lines: List[str], key_re, offset=1, types=None, end_characters=None
    ):
        """
        A parser to parse blocks of data by searching a title line
        Example:
            HEADER  <- header line used for matching
            XXXX       ^
            ---------  |
            A 1 B 2    | Data starts here so offset is 3
            A 1 B 2
            C 1 D 2
            ---------  <- data ends here so the end character is "-----" (default)

        :param lines: A list contains string of each line
        :param key_re: The regular expression to match the presence of the block
        :param offset: Offset from the header to the real data
        :param types: The types of the data for each line
        """
        self.lines = lines
        self.key_re = key_re
        self.types = types
        self.offset = offset
        self.blocks = []
        self.end_characters = [] if not end_characters else end_characters
        self.end_characters += self.DEFAULT_END_CHAR

    def parse(self):
        """Parse the data"""
        for i, line in enumerate(self.lines):
            m = self.key_re.match(line)
            # not matching a header line
            if m is None:
                continue
            # We have found a matched group
            block_name = m.groups()
            block_tokens = []
            j = i + self.offset
            while j <= len(self.lines):
                this_line = self.lines[j].strip()
                # Break with empty line
                if not this_line:
                    break
                # Break with predefined sequence such as ---- or +++++
                if any(key in line for key in self.end_characters):
                    break
                tokens = self.lines[j].strip().split()
                block_tokens.append(tokens)
                j += 1
            self.blocks.append((block_name, block_tokens))

        if self.types is not None:
            self.blocks = self.convert_type()
        return self.blocks

    def convert_type(self):
        """Convert the match data to the correct type"""
        assert self.blocks
        converted = []
        for block_name, block_tokens in self.blocks:
            new_block = []
            for tokens in block_tokens:
                # Use the type constructors to convert the string data to the right type
                new_block.append(
                    [
                        constructor(token)
                        for constructor, token in zip(self.types, tokens)
                    ]
                )
            converted.append([block_name, new_block])
        return converted


class BandsParser(BaseRawParser):
    """Parser to process the BNADS_XX.dat files"""

    def parse(self):
        """Parse the bands.dat file"""
        arrays = []
        for line in self.lines:
            if not line:
                continue
            arrays.append(np.fromstring(line, sep=" ", dtype=float))
        data = np.stack(arrays, axis=0)[:, 1:]
        kdist = data[:, 0]
        eigenvalues = data[:, 1:]
        return kdist, eigenvalues


class KpointsParser(BaseRawParser):
    """
    Parse the kpoints file in the suffix.out folder
    """

    def parse(self):
        """Read the output kpoints file"""

        line = self.lines[0]
        nkpts = int(line.strip().split()[-1])
        assert self.lines[1].startswith("K-POINTS DIRECT COORDINATES")
        points = []
        weights = []
        for i in range(nkpts):
            tokens = self.lines[i + 3].strip().split()
            points.append([float(tokens[i]) for i in range(1, 4)])
            weights.append(float(tokens[4]))
        return points, weights


class InternalParametersParser(BaseRawParser):
    """
    Parse the INPUT file in the suffix.out folder
    NOTE: This does not work for a general INPUT file
    """

    def parse(self):
        """Read the output internal parameters file"""
        out_dict = {}
        # Skip the first line
        for _line in self.lines[1:]:
            if _line.startswith("#"):
                continue
            line = _line.strip()
            if not line:
                continue
            # Remove the trialing # comments
            match = re.match(r"^(.+) *#.*$", line)
            if match:
                tokens = match.group(1).split(maxsplit=1)
            else:
                tokens = line.split(maxsplit=1)
            # Add potential null value
            if len(tokens) != 2:
                tokens.append("None")
            out_dict[tokens[0].strip()] = tokens[1].strip()
        return out_dict


class StruParser(BaseRawParser):
    """
    Parse a STRU file
    """

    def parse(self):
        """
        Parse a STRU file
        :returns: A tuple of lattice vectors, positions, species.
        """
        blocks = self.parse_blocks()
        # LATTICE_CONSTANT may carry a trailing comment; v3.11 STRU_FINAL
        # writes "1.889... # in Bohr (= 1 Angstrom); lattice vectors below
        # are in Angstrom", meaning the vectors are already in Angstrom.
        lat_line = blocks["LATTICE_CONSTANT"][0]
        lattice_constant = float(lat_line.split("#")[0].split()[0])  # In bohr
        # v3.11 STRU_FINAL writes the constant in bohr-per-angstrom
        # (1.8897...) with the lattice vectors already in Angstrom;
        # v3.10 wrote the constant in bohr with unitless vectors.
        vectors_in_angstrom = abs(lattice_constant - 1.8897268778) < 1e-3
        lattice_vectors = np.array(
            [
                [float(value) for value in line.split()]
                for line in blocks["LATTICE_VECTORS"]
            ]
        )
        positions = []
        species = []
        # magnetic_moments = []
        pos_block = blocks["ATOMIC_POSITIONS"]
        coord_type = pos_block[0]
        current_specie = None
        p = 1
        while p < len(pos_block):
            current_specie = pos_block[p]
            # current_magmom = float(pos_block[p+1])
            current_natoms = float(pos_block[p + 2])
            for i in range(int(current_natoms)):
                tokens = pos_block[p + 3 + i].split()
                positions.append([float(value) for value in tokens[:3]])
                species.append(current_specie)
            p += 3 + int(current_natoms)
        positions = np.array(positions)
        # Lattice vectors in angstrom
        if not vectors_in_angstrom:
            lattice_vectors *= lattice_constant / 1.8897261255

        if coord_type == "Direct":
            positions = positions @ lattice_vectors
        elif coord_type == "Cartesian":
            positions *= lattice_constant / 1.8897261255
        elif coord_type == "Cartesian_au":
            positions *= 1.0 / 1.8897261255
        elif coord_type == "Cartesian_angstrom":
            pass
        else:
            raise ValueError(f"Unknown coordinate type {coord_type}")
        self.structure = {
            "lattice_vectors": lattice_vectors,
            "species": species,
            "positions": positions,
        }
        return lattice_vectors, positions, species

    def parse_structure(self):
        """Parse for the structure"""
        return self.parse()

    def parse_blocks(self):
        """Split the file content by their blocks"""
        keywords = [
            "ATOMIC_SPECIES",
            "LATTICE_CONSTANT",
            "LATTICE_VECTORS",
            "ATOMIC_POSITIONS",
        ]
        blocks = {}
        current_block = None
        for _line in self.lines:
            line = _line.strip()
            # Skip comment lines
            if not line or line.startswith("#"):
                continue
            # Remove any trailing comments
            line = line.split("#", maxsplit=1)[0].strip()
            # Check if we are in a block title line
            is_title = False
            for block_name in keywords:
                if block_name in line:
                    current_block = block_name
                    blocks[current_block] = []
                    is_title = True
                    continue
            if is_title:
                continue
            # We are in a block - record the content
            if current_block is not None:
                blocks[current_block].append(line)
        self.blocks = blocks
        return blocks
