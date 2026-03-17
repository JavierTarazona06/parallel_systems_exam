# Simulation d'une galaxie à n corps en utilisant une grille spatiale pour accélérer le calcul des forces gravitationnelles.
#     On crée une classe représentant le système de corps avec la méthode d'intégration basée sur une grille.
# On utilise numba pour accélérer les calculs.
import numpy as np
import visualizer3d
import sys
from numba import njit, prange
from mpi4py import MPI

# Unités:
# - Distance: année-lumière (ly)
# - Masse: masse solaire (M_sun)
# - Vitesse: année-lumière par an (ly/an)
# - Temps: année

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13
GHOST_CELL_WIDTH = 2

def generate_star_color(mass : float) -> tuple[int, int, int]:
    """
    Génère une couleur pour une étoile en fonction de sa masse.
    Les étoiles massives sont bleues, les moyennes sont jaunes, les petites sont rouges.
    
    Parameters:
    -----------
    mass : float
        Masse de l'étoile en masses solaires
    
    Returns:
    --------
    color : tuple
        Couleur RGB (R, G, B) avec des valeurs entre 0 et 255
    """
    if mass > 5.0:
        # Étoiles massives: bleu-blanc
        return (150, 180, 255)
    elif mass > 2.0:
        # Étoiles moyennes-massives: blanc
        return (255, 255, 255)
    elif mass >= 1.0:
        # Étoiles comme le Soleil: jaune
        return (255, 255, 200)
    else:
        # Étoiles de faible masse: rouge-orange
        return (255, 150, 100)

@njit(parallel=True)
def update_stars_in_grid( cell_start_indices : np.ndarray, body_indices : np.ndarray,
                          cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                          masses: np.ndarray,
                          positions : np.ndarray, grid_min : np.ndarray, grid_max : np.ndarray,
                          cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    # Réinitialise les compteurs de début des cellules
    cell_start_indices.fill(-1)
    # Compte le nombre de corps dans chaque cellule
    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        # Gère le cas où un corps est exactement sur la borne max   
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        cell_counts[morse_idx] += 1
    # Calcule les indices de début des cellules
    running_index = 0
    for i in range(len(cell_counts)):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[len(cell_counts)] = running_index # Fin du dernier corps
    # Remplit les indices des corps dans les cellules
    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1
    # Maintenant, on peut calculer le centre de masse et la masse totale de chaque cellule
    for i in prange(len(cell_counts)):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i+1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody] 
            cell_mass += m
            com_position += positions[ibody] * m
        if cell_mass > 0.0:
            com_position /= cell_mass
        # Stocke les résultats dans des tableaux globaux
        cell_masses[i] = cell_mass
        cell_com_positions[i] = com_position

@njit(parallel=True)
def compute_acceleration( positions : np.ndarray, masses : np.ndarray,
                          cell_start_indices : np.ndarray, body_indices : np.ndarray,
                          cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                          grid_min : np.ndarray, grid_max : np.ndarray,
                          cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    a = np.zeros_like(positions)
    for ibody in prange(n_bodies):
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        # Parcourt toutes les cellules pour calculer la contribution gravitationnelle
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy*n_cells[0] + iz*n_cells[0]*n_cells[1]
                    if (abs(ix-cell_idx[0]) > 2) or (abs(iy-cell_idx[1]) > 2) or (abs(iz-cell_idx[2]) > 2):
                        cell_com = cell_com_positions[morse_idx]    
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - pos
                            distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                            if distance > 1.E-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a[ibody,:] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        # Parcourt les corps dans cette cellule
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx+1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                                if distance > 1.E-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    a[ibody,:] += G * direction[:] * inv_dist3 * masses[jbody]
    return a

# On crée une grille cartésienne régulière pour diviser l'espace englobant la galaxie en cellules
class SpatialGrid:
    """_summary_
    """
    def __init__(self, positions : np.ndarray, nb_cells_per_dim : tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.n_cells = np.array(nb_cells_per_dim)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        # On va stocker les indices des corps dans chaque cellule adéquate
        # Les cellules seront stockées sous une forme morse : indice de la cellule = ix + iy*n_cells_x + iz*n_cells_x*n_cells_y
        # et on gère deux tableaux : un pour le début des indices de chaque cellule, un autre pour les indices des corps
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        # Stockage du centre de masse de chaque cellule et de la masse totale contenue dans chaque cellule
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)
        
    def update_bounds(self, positions : np.ndarray):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        
    def update(self, positions : np.ndarray, masses : np.ndarray):
        #self.update_bounds(positions)
        update_stars_in_grid( self.cell_start_indices, self.body_indices,                             
                              self.cell_masses, self.cell_com_positions,
                              masses,
                              positions, self.min_bounds, self.max_bounds,
                              self.cell_size, self.n_cells)

class NBodySystem:
    def __init__(self, filename, ncells_per_dir : tuple[int, int, int] = (10,10,10)):
        positions = []
        velocities = []
        masses    = []
        
        self.max_mass = 0.
        self.box = np.array([[-1.E-6,-1.E-6,-1.E-6],[1.E-6,1.E-6,1.E-6]], dtype=np.float64) # Contient les coins min et max du système
        with open(filename, "r") as fich:
            line = fich.readline() # Récupère la masse, la position et la vitesse sous forme de chaîne
            # Récupère les données numériques pour instancier un corps qu'on rajoute aux corps déjà présents :
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])
                
                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i]-1.E-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i]+1.E-6)
                    
                line = fich.readline()
        
        self.positions  = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses     = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)
        
    def update_positions(self, dt):
        """Applique la méthode de Verlet vectorisée pour mettre à jour les positions et vitesses des corps."""
        a = compute_acceleration( self.positions, self.masses,
                                  self.grid.cell_start_indices, self.grid.body_indices,
                                  self.grid.cell_masses, self.grid.cell_com_positions,
                                  self.grid.min_bounds, self.grid.max_bounds,
                                  self.grid.cell_size, self.grid.n_cells)
        self.positions += self.velocities * dt + 0.5 * a * dt * dt
        self.grid.update(self.positions, self.masses)
        a_new = compute_acceleration( self.positions, self.masses,
                                      self.grid.cell_start_indices, self.grid.body_indices,
                                      self.grid.cell_masses, self.grid.cell_com_positions,
                                      self.grid.min_bounds, self.grid.max_bounds,
                                      self.grid.cell_size, self.grid.n_cells)
        self.velocities += 0.5 * (a + a_new) * dt

system : NBodySystem

def update_positions(dt : float):
    global system
    system.update_positions(dt)
    return system.positions


def create_visualizer(system: NBodySystem):
    pos = system.positions
    col = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    bounds = [
        [system.box[0][0], system.box[1][0]],
        [system.box[0][1], system.box[1][1]],
        [system.box[0][2], system.box[1][2]],
    ]
    return visualizer3d.Visualizer3D(pos, col, intensity, bounds)


def split_cells_along_x(n_cells_x: int, size: int, rank: int) -> tuple[int, int]:
    base = n_cells_x // size
    remainder = n_cells_x % size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def compute_x_cell_indices(positions: np.ndarray, grid_min: np.ndarray, cell_size: np.ndarray, n_cells: np.ndarray) -> np.ndarray:
    cell_x = np.floor((positions[:, 0] - grid_min[0]) / cell_size[0]).astype(np.int64)
    return np.clip(cell_x, 0, n_cells[0] - 1)


def select_local_and_owned_bodies(system: NBodySystem, rank: int, size: int):
    x_start, x_end = split_cells_along_x(system.grid.n_cells[0], size, rank)
    ghost_start = max(0, x_start - GHOST_CELL_WIDTH)
    ghost_end = min(system.grid.n_cells[0], x_end + GHOST_CELL_WIDTH)

    cell_x = compute_x_cell_indices(
        system.positions,
        system.grid.min_bounds,
        system.grid.cell_size,
        system.grid.n_cells,
    )

    owned_mask = (cell_x >= x_start) & (cell_x < x_end)
    local_mask = (cell_x >= ghost_start) & (cell_x < ghost_end)

    local_global_indices = np.nonzero(local_mask)[0].astype(np.int64)
    owned_global_indices = np.nonzero(owned_mask)[0].astype(np.int64)
    owned_local_indices = np.nonzero(owned_mask[local_mask])[0].astype(np.int64)

    local_positions = system.positions[local_global_indices].copy()
    local_velocities = system.velocities[local_global_indices].copy()
    local_masses = system.masses[local_global_indices].copy()

    owned_positions = system.positions[owned_global_indices].copy()
    owned_velocities = system.velocities[owned_global_indices].copy()
    owned_masses = system.masses[owned_global_indices].copy()

    return (
        local_global_indices,
        owned_global_indices,
        owned_local_indices,
        local_positions,
        local_velocities,
        local_masses,
        owned_positions,
        owned_velocities,
        owned_masses,
    )


def build_grid_arrays(grid: SpatialGrid, positions: np.ndarray, masses: np.ndarray):
    cell_start_indices = np.full(np.prod(grid.n_cells) + 1, -1, dtype=np.int64)
    body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
    cell_masses = np.zeros(shape=(np.prod(grid.n_cells),), dtype=np.float32)
    cell_com_positions = np.zeros(shape=(np.prod(grid.n_cells), 3), dtype=np.float32)

    update_stars_in_grid(
        cell_start_indices,
        body_indices,
        cell_masses,
        cell_com_positions,
        masses,
        positions,
        grid.min_bounds,
        grid.max_bounds,
        grid.cell_size,
        grid.n_cells,
    )

    return cell_start_indices, body_indices, cell_masses, cell_com_positions


def compute_global_cell_data(system: NBodySystem, owned_positions: np.ndarray, owned_masses: np.ndarray, comm):
    _, _, local_cell_masses, local_cell_com_positions = build_grid_arrays(system.grid, owned_positions, owned_masses)
    local_weighted_positions = local_cell_com_positions * local_cell_masses[:, np.newaxis]

    global_cell_masses = np.zeros_like(local_cell_masses)
    global_weighted_positions = np.zeros_like(local_weighted_positions)

    comm.Allreduce(local_cell_masses, global_cell_masses, op=MPI.SUM)
    comm.Allreduce(local_weighted_positions, global_weighted_positions, op=MPI.SUM)

    global_cell_com_positions = np.zeros_like(local_cell_com_positions)
    non_empty = global_cell_masses > 0.0
    global_cell_com_positions[non_empty] = (
        global_weighted_positions[non_empty] / global_cell_masses[non_empty, np.newaxis]
    )

    return global_cell_masses, global_cell_com_positions


def synchronize_owned_values(global_array: np.ndarray, owned_indices: np.ndarray, owned_values: np.ndarray, comm):
    gathered_updates = comm.allgather((owned_indices, owned_values))
    for indices, values in gathered_updates:
        if len(indices) > 0:
            global_array[indices] = values


def distributed_update_positions(system: NBodySystem, dt: float, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    (
        local_global_indices,
        owned_global_indices,
        owned_local_indices,
        local_positions,
        local_velocities,
        local_masses,
        owned_positions,
        owned_velocities,
        owned_masses,
    ) = select_local_and_owned_bodies(system, rank, size)

    global_cell_masses, global_cell_com_positions = compute_global_cell_data(system, owned_positions, owned_masses, comm)
    local_cell_start_indices, local_body_indices, _, _ = build_grid_arrays(system.grid, local_positions, local_masses)
    local_accelerations = compute_acceleration(
        local_positions,
        local_masses,
        local_cell_start_indices,
        local_body_indices,
        global_cell_masses,
        global_cell_com_positions,
        system.grid.min_bounds,
        system.grid.max_bounds,
        system.grid.cell_size,
        system.grid.n_cells,
    )
    owned_accelerations = local_accelerations[owned_local_indices].copy()
    global_accelerations = np.zeros_like(system.positions)
    synchronize_owned_values(global_accelerations, owned_global_indices, owned_accelerations, comm)

    owned_positions += owned_velocities * dt + 0.5 * owned_accelerations * dt * dt
    synchronize_owned_values(system.positions, owned_global_indices, owned_positions, comm)

    (
        local_global_indices,
        owned_global_indices,
        owned_local_indices,
        local_positions,
        local_velocities,
        local_masses,
        owned_positions,
        owned_velocities,
        owned_masses,
    ) = select_local_and_owned_bodies(system, rank, size)

    global_cell_masses, global_cell_com_positions = compute_global_cell_data(system, owned_positions, owned_masses, comm)
    local_cell_start_indices, local_body_indices, _, _ = build_grid_arrays(system.grid, local_positions, local_masses)
    local_accelerations_new = compute_acceleration(
        local_positions,
        local_masses,
        local_cell_start_indices,
        local_body_indices,
        global_cell_masses,
        global_cell_com_positions,
        system.grid.min_bounds,
        system.grid.max_bounds,
        system.grid.cell_size,
        system.grid.n_cells,
    )
    owned_accelerations_new = local_accelerations_new[owned_local_indices].copy()

    owned_accelerations_old = global_accelerations[owned_global_indices].copy()
    owned_velocities += 0.5 * (owned_accelerations_old + owned_accelerations_new) * dt
    synchronize_owned_values(system.velocities, owned_global_indices, owned_velocities, comm)

    system.grid.cell_masses[:] = global_cell_masses
    system.grid.cell_com_positions[:] = global_cell_com_positions

def run_simulation(filename, geometry=(800,600), ncells_per_dir : tuple[int, int, int] = (10,10,10), dt=0.001):
    global system
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("Exécution séquentielle : lancer avec au moins 2 processus MPI pour séparer affichage et calcul.")
            system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
            visu = create_visualizer(system)
            visu.run(updater=update_positions, dt=dt)
        return

    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)

    visu = None
    if rank == 0:
        visu = create_visualizer(system)
        print("Contrôles :")
        print("  - Clic gauche + déplacement souris : rotation de la caméra")
        print("  - Molette de la souris : zoom")
        print("  - ESC ou fermeture de fenêtre : quitter")

    running = True
    while True:
        if rank == 0:
            running = visu._handle_events()
        running = comm.bcast(running, root=0)
        if not running:
            break

        distributed_update_positions(system, dt, comm)

        if rank == 0:
            visu.update_points(system.positions)
            visu._render()

    if rank == 0 and visu is not None:
        visu.cleanup()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    filename = "data/galaxy_1000"
    dt = 0.001
    n_cells_per_dir = (20, 20, 1)

    if len(argv) > 0:
        filename = argv[0]
    if len(argv) > 1:
        dt = float(argv[1])
    if len(argv) > 4:
        n_cells_per_dir = (int(argv[2]), int(argv[3]), int(argv[4]))

    print(f"Simulation de {filename} avec dt = {dt} et grille {n_cells_per_dir}")
    run_simulation(filename, ncells_per_dir=n_cells_per_dir, dt=dt)


if __name__ == "__main__":
    main()
    
