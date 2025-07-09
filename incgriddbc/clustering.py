import math

class IncrementalClustering:
    def __init__(self, eps, min_pts, dim):
        self.eps = eps
        self.min_pts = min_pts
        self.dim = dim
        
        self.X = [] # Input
        self.y = [] # Labels
        self.p = [] # Property 1: Core 0: Border -1: Noise
        self.sparse_grid = {}  
        self.data_grid = {}
        self.query_count = 0
        
    
        # self.create_grid()
        
    def create_grid(self):
        self.cell_num = 2
        self.cell_len = 0.5
        
        cell_thr = self.eps / math.sqrt(self.dim)

        while cell_thr < self.cell_len:
            self.cell_num *= 2
            self.cell_len = 1 / self.cell_num

        self.key_len = len(str(self.cell_num))
        self.nei_step = math.ceil(self.eps/self.cell_len) 
                
        print(f"Grid Construction- Eps: {self.eps} MP: {self.min_pts} Dim: {self.dim} CN: {self.cell_num} CL: {self.cell_len} NS: {self.nei_step} PG: {math.pow(self.cell_num, self.dim)} ")

        
    def add_cell(self, cell):
        
        # Add a cell
        current_level = self.sparse_grid
        for coord in cell[:-1]:
            current_level = current_level.setdefault(coord, {})
            
      
        if cell[-1] not in current_level:
            current_level[cell[-1]] = [[], []]
        
            # Find neighbors
            neighbors = self.get_neighbor_cells(cell)
            
            for neighbor in neighbors:
                mid, mad = self.calculate_cell_distance(cell, neighbor)
                
                if mad <= self.eps:
                    current_level[cell[-1]][0].append(neighbor)
                    self.add_neighbor_cell(neighbor, cell, full=True)
                elif mid <= self.eps:
                    current_level[cell[-1]][1].append(neighbor)
                    self.add_neighbor_cell(neighbor, cell, full=False)
                # else:
                #     print(self.eps, mid, mad, cell, neighbor)
    
        else:
            neighbors = current_level[cell[-1]]
            
        return neighbors
        

    def get_neighbor_cells(self, cell):
        neighbors = []
        
        def recursive_check(level, coords, depth):
            if depth == len(cell): 
                if coords != cell:
                    neighbors.append(coords)
                return

            for offset in range(-self.nei_step, self.nei_step + 1):
                new_coord = cell[depth] + offset
                if new_coord in level:
                    recursive_check(level[new_coord], coords + (new_coord,), depth + 1)
        
        recursive_check(self.sparse_grid, (), 0)

        return neighbors

    def add_neighbor_cell(self, cell, neighbor, full=True):
        current_level = self.sparse_grid
        for coord in cell[:-1]:
            current_level = current_level.setdefault(coord, {})
        
        if cell[-1] in current_level:
            if neighbor not in current_level[cell[-1]]:
                if full:
                    current_level[cell[-1]][0].append(neighbor)
                else:
                    current_level[cell[-1]][1].append(neighbor)

    
    def get_reachable_points(self, point, target_cell):
        reachable_points = self.data_grid.get(target_cell, []).copy()
        
        candidate_cells = self.add_cell(target_cell)
        
        for candidate_cell in candidate_cells[0]:
            reachable_points.extend(self.data_grid.get(candidate_cell, []))
            
        for candidate_cell in candidate_cells[1]:
            candidate_points = self.data_grid.get(candidate_cell, [])
            for candidate_point in candidate_points:
                dist = math.dist(point, self.X[candidate_point])
                self.query_count += 1
                if dist<=self.eps:
                    reachable_points.append(candidate_point)
                
        return reachable_points
        
                
    def add_points(self, points):
        last_point_index = len(self.X)
        
        for i, point in enumerate(points):
            target_cell, _ = self.get_location(point)
            
            self.add_cell(target_cell)
            self.X.append(point)
            self.y.append(-2)
            self.p.append(-1)
            
            self.data_grid.setdefault(target_cell, []).append(last_point_index + i)
        
        
        for i, point in enumerate(points): 
            index = last_point_index + i          
            y_ = self.y[index]
            p_ = self.p[index]
            
            if y_ == -2:
                target_cell, _ = self.get_location(point)
                
                self.query_count = 0
                reachable_points = self.get_reachable_points(point, target_cell)
                
                rp_dict = {}
                
                for rp in reachable_points:
                    if rp == index: continue
                    y__ = self.y[rp]
                    p__ = self.p[rp]
                    
                    if y__ == -2:
                        rp_dict.setdefault(-1, {}).setdefault(y__, []).append(rp)
                    else:
                        rp_dict.setdefault(p__, {}).setdefault(y__, []).append(rp)
                    
                if len(reachable_points) >= self.min_pts:
                    p_  = 1
                    self.p[index] = p_
                    
                k = -1
                if 1 in rp_dict:
                    core_labels = list(rp_dict.get(1, {}).keys())
                    k = min(core_labels)
                    
                    # Case Absorption
                    self.y[index] = k
                    
                    if len(core_labels) >= 2 and p_ == 1:
                        # Case Merging
                        for cl in core_labels:
                            if k == cl: continue
                            self.y = [k if x == cl else x for x in self.y]
                else:
                    if p_ == -1:
                        # Case Noise
                        self.y[index] = k
                    else:
                        # Case Creation
                        k = max(self.y) + 1
                        k = 0 if k < 0 else k
                        
                        # print("Creation", k)
                        self.y[index] = k
                     
                if k > -1:
                    noise_dict = rp_dict.get(-1, {})
                    
                    for ke, rp_noises in noise_dict.items():
                        for rp in rp_noises:
                            self.y[rp] = k
                            self.p[rp] = 0
                            
                            rp_dict = self.move_object(rp_dict, (-1, ke), (0, k), rp)
                
                
                    
                # Label Propagation
                border_dict = rp_dict.get(0, {})
                labels = sorted(list(border_dict.keys()))
                
                for k in labels:
                    rp_borders = border_dict.get(k, [])
                    for rp in rp_borders:
                        if rp == index: continue
                        self.update_clusters(index, rp)
                    
                noise_dict = rp_dict.get(-1, {})
                rp_noises = noise_dict.get(-1, [])
                for rp in rp_noises:
                    if rp == index: continue
                    self.update_clusters(index, rp)
                
                
    def update_clusters(self, np, up):
        point = self.X[up]
        
        p_ = self.p[up]
        
        target_cell, _ = self.get_location(point)
        
        reachable_points = self.get_reachable_points(point, target_cell)
        
        rp_dict = {}
        for rp in reachable_points:
            if up == rp: continue
            y__ = self.y[rp]
            p__ = self.p[rp]
            
            if y__ == -2:
                rp_dict.setdefault(-1, {}).setdefault(y__, []).append(rp)
            else:
                rp_dict.setdefault(p__, {}).setdefault(y__, []).append(rp)
        
        if len(reachable_points) >= self.min_pts:
            p_ = 1
            self.p[up] = p_
        
        
        k = -1
        if 1 in rp_dict:
            core_labels = list(rp_dict.get(1, {}).keys())
            k = min(core_labels)
            
            self.y[up] = k
            
            if len(core_labels) >= 2 and p_ == 1:
                for cl in core_labels:
                    if k == cl: continue
                    self.y = [k if x == cl else x for x in self.y]
        else:
            if p_ == 1:
                
                k = max(self.y) + 1
                k = 0 if k < 0 else k
                
                # Creation
                self.y[up] = k
            else:
                self.y[up] = k
                
        
        if k > -1 and p_ == 1:
            rp_noises =  rp_dict.get(-1, {}).get(-1, [])
            
            for rp in rp_noises:
                if rp == up: continue
                self.y[rp] = k
                self.p[rp] = 0
                    
       
                
    def get_location(self, point):
        location = [math.floor(p/self.cell_len) for p in point]
        location = tuple([min(self.cell_num-1, l) for l in location])
        address = ''.join([str(l).zfill(self.key_len) for l in location])

        return location, address
    
    
    def calculate_cell_distance(self, cell1, cell2):
        dist_min = 0 
        dist_max = 0
        
        for i, j in zip(cell1, cell2):
            dist_min += ((abs(i-j)-1)*self.cell_len) ** 2
            dist_max += ((abs(i-j)+1)*self.cell_len) ** 2
            
        
        dist_min = math.sqrt(dist_min)
        dist_max = math.sqrt(dist_max)
        
        
        return dist_min, dist_max
    
    def move_object(self, nested_dict, from_key, to_key, obj):
        from_list = nested_dict[from_key[0]][from_key[1]]
        
        if to_key[0] not in nested_dict:
            nested_dict[to_key[0]] = {}
        if to_key[1] not in nested_dict[to_key[0]]:
            nested_dict[to_key[0]][to_key[1]] = []
        to_list = nested_dict[to_key[0]][to_key[1]]
        
        if obj in from_list:
            from_list.remove(obj)
        
        to_list.append(obj)

        return nested_dict
            
        
    
if __name__ == '__main__':
    import random
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.cluster import normalized_mutual_info_score
    from sklearn.metrics.cluster import adjusted_rand_score

    random.seed(1)
    np.random.seed(1)
    
    
    points = random_points = np.random.rand(8000, 2)
    print(points.shape)
    

    eps, mp = 0.4, 6
    dims = points.shape[1]
    
    
    ic = IncrementalClustering(eps, mp, dims)
    
    # print(points1[0])
    
    
    
    
    # for point in points.tolist():
    #     ic.add_points([point])
        
    ic.add_points(points.tolist())
    
    print("=======================")
    
    print(ic.query_count)
    # print(ic.p)
    
    # print(grids[0], grids[0],  ic.calculate_cell_distance((2,2), (2,2)))
    
    
    clustering = DBSCAN(eps=eps, min_samples=mp).fit(points)
    # print(list(clustering.labels_), type(clustering.labels_))
    # # print(clustering.core_sample_indices_)
    print("=======================")
    print("NMI", normalized_mutual_info_score(ic.y, clustering.labels_))
    print("ARI", adjusted_rand_score(ic.y, clustering.labels_))
    
