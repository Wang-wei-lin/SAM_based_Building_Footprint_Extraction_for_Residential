from feature_extractor import FeatureExtractorV2
import shapely
from rasterio import features
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box
from skimage.segmentation import slic
from skimage.graph._rag import RAG
from rasterio.mask import mask
import laspy


def cluster2d(shp_fp, save_fp, eps=1.5, min_samples=10):
    """
    基于点矢量文件进行二维密度聚类，方法采用dbscan，聚类结果保存成点矢量shp文件，”cluster“属性为聚类类别(Perform 2D density clustering based on
    point vector file, using dbscan method, save clustering result as point vector shp file, "cluster" attribute is clustering category)
    :param shp_fp: 点矢量文件路径(point vector file path)
    :param save_fp: 结果保存路径(result save path)
    :param eps: 邻域半径，用于定义一个点的邻域范围(neighborhood radius, used to define the neighborhood range of a point)
    :param min_samples: 一个区域内必须包含的最小样本数(minimum number of samples that must be included in an area)
    :return:
    """
    point_gdf = gpd.read_file(shp_fp)
    X = point_gdf[['geometry']].apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1).tolist()
    X = [list(i) for i in X]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)
    point_gdf['cluster'] = clusters
    point_gdf.to_file(save_fp)


def cal_convex_hull(clustered_shp_fp: str, save_fp: str, point_threshold=100, area_threshold=50, merged=True):
    """
    基于分层聚类后点云数据，为每一类点云数据构建凸包多边形，并进一步通过点数阈值以及面积阈值过滤数据，最终结果为粗糙建筑物近似面矢量。(Based on the hierarchical
    clustering point cloud data, a convex hull polygon is constructed for each type of point cloud data, and the data is further filtered
     by the point count threshold and area threshold. The final result is a rough building approximate surface vector.)
    :param clustered_shp_fp:聚类后点云数据路径(point cloud data path after clustering)
    :param save_fp: 凸包多边形结果保存路径(convex hull polygon result save path)
    :param point_threshold:  点数阈值，用于过滤点数点少的聚类类别(point count threshold, used to filter cluster categories with few points)
    :param area_threshold:  面积阈值，用于过滤面积较小的凸包多边形(area threshold, used to filter convex hull polygons with smaller areas)
    :param merged:  是否融合重叠的凸包多边形矢量，默认为是(whether to merge overlapping convex hull polygon vectors, the default is yes)
    :return:
    """
    clustered_gdf = gpd.read_file(clustered_shp_fp)

    result = []
    cluster_list = []
    mean_z_list = []
    cluster_attr = clustered_gdf['cluster']
    cluster_uni = np.unique(cluster_attr[cluster_attr >= 0]).tolist()

    for cluster in cluster_uni:
        process_gdf = clustered_gdf[clustered_gdf['cluster'] == cluster].copy()
        if len(process_gdf) > point_threshold:
            # print(f"processing: {cluster}")
            coords = process_gdf[['X', 'Y']].values
            Z = process_gdf['Z'].values
            mean_z = np.mean(Z)

            hull = ConvexHull(coords)
            convex_hull_indices = hull.vertices
            convex_hull_coords = coords[convex_hull_indices]
            convex_hull_polygon = Polygon(convex_hull_coords)

            if convex_hull_polygon.area > area_threshold:
                result.append(convex_hull_polygon)
                cluster_list.append(cluster)
                mean_z_list.append(mean_z)

    df = pd.DataFrame(data=np.vstack([cluster_list, mean_z_list]).T, columns=['cluster', "mean_z"])
    df['mean_z'] = df['mean_z'].astype(float)
    gdf = gpd.GeoDataFrame(data=df, geometry=result, crs=clustered_gdf.crs)

    if merged:
        dissolved_gdf = gdf.dissolve()
        out_gdf = dissolved_gdf.explode(ignore_index=True)
        out_gdf['build_idx'] = [i for i in range(len(out_gdf))]
        out_gdf.to_file(save_fp)
    else:
        gdf.to_file(save_fp)


def create_prompt_via_centroid(merged_gdf):
    """
    基于建筑物近似面的质心构建单点提示(Construct single point prompts based on the centroid of the building approximate surface)
    :param merged_gdf: 建筑物近似面数据(Building approximate surface data)
    :return: 单点提示(Single point prompt)
    """
    centroid_gs = merged_gdf.centroid
    centroid_gdf = gpd.GeoDataFrame(geometry=centroid_gs, crs=merged_gdf.crs)
    """indices = [i for i in range(len(centroid_gdf))]
    centroid_gdf['index'] = indices"""
    centroid_gdf['index'] = merged_gdf['build_idx']
    return centroid_gdf


def create_prompt_via_grid(merged_gdf):
    """
    基于平均布点构建多点提示(Construct multiple point prompts based on average point distribution)
    :param merged_gdf: 建筑物近似面数据(approximate building surface data)
    :return: 多点提示(multiple point prompts)
    """
    indices = []
    gdf_list = []
    step = 1
    bounds = merged_gdf.bounds
    for i in merged_gdf['build_idx']:
        i = int(i)
        geom = merged_gdf.loc[i, "geometry"]
        bound = bounds.loc[i]
        x_range = np.arange(bound['minx'], bound['maxx'], step)
        y_range = np.arange(bound['miny'], bound['maxy'], step)
        x_coords, y_coords = np.meshgrid(x_range, y_range)
        points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=x_coords.flatten(), y=y_coords.flatten()))
        its_gdf = points_gdf[points_gdf.intersects(geom)]
        gdf_list.append(its_gdf)
        indices += [i] * len(its_gdf)

    evenly_gdf = pd.concat(gdf_list)
    evenly_gdf['index'] = indices
    evenly_gdf.set_crs(merged_gdf.crs, inplace=True)
    return evenly_gdf


def create_prompt_via_bound(merged_gdf):
    """
    基于bounding_box构建框提示(Build box hint based on bounding_box)
    :param merged_gdf: 建筑物近似面数据(approximate building surface data)
    :return: 框提示，每个建筑物对应一个框(box hint, each building corresponds to a box)
    """
    bounding_boxes = []
    bounds = merged_gdf.bounds
    for i in merged_gdf['build_idx']:
        i = int(i)
        bound = bounds.loc[i]
        bounding_box = box(bound['minx'], bound['miny'], bound['maxx'], bound['maxy'])
        bounding_boxes.append(bounding_box)
    bbox_gdf = gpd.GeoDataFrame(geometry=bounding_boxes, crs=merged_gdf.crs)
    indices = [i for i in range(len(bbox_gdf))]
    bbox_gdf['index'] = indices
    return bbox_gdf


def create_prompt_via_sample(merged_gdf, point_gdf, sample_method="centroid_random"):
    """
    基于采样构建点提示，目前实现随机采样√、随机质心采样√、均匀采样√(Based on sampling, point prompts are constructed. Currently,
    random sampling√, random centroid sampling√, and uniform sampling√ are implemented.)
    :param merged_gdf: 建筑物近似面数据(approximate building surface data)
    :param sample_method: 采样方法，默认为centroid_random,可选方法:"random"、"uniform"(sampling method, default is centroid_random,
     optional methods: "random", "uniform")
    :return: 存放提示的gdf(gdf where the prompt is stored)
    """
    sindex = point_gdf.sindex
    indices = []
    points_list = []
    for i in merged_gdf['build_idx']:
        i = int(i)
        geom = merged_gdf.loc[i, 'geometry']
        geom_contract = geom.buffer(-1)

        possible_matches_index = sindex.intersection(geom_contract.bounds)
        possible_matches = point_gdf.iloc[possible_matches_index]
        its_points = possible_matches[possible_matches.intersects(geom_contract)]
        its_points.reset_index(inplace=True, drop=True)

        # 随机采样
        # Random sampling
        if sample_method == "random":
            np.random.seed(0)
            num_points = 10
            random_indices = np.random.randint(0, len(its_points), num_points)
            random_points = its_points.loc[random_indices]
            points_list.append(random_points)
            indices += [i] * len(random_points)

        # 质心-随机采样
        # Centroid - Random Sampling
        elif sample_method == "centroid_random":
            centroid = geom.centroid
            np.random.seed(0)
            num_points = 10
            random_indices = np.random.randint(0, len(its_points), num_points-1)
            random_points = its_points.loc[random_indices]
            random_points.loc[len(random_points), 'geometry'] = centroid
            points_list.append(random_points)
            indices += [i] * len(random_points)

        # 均匀采样
        # Uniform sampling
        elif sample_method == 'uniform':
            sample_step, points_per_step = 5, 3
            centroid = geom.centroid
            sample_list = []
            for r in range(0, 1000000000000000, sample_step):  # while True
                print(r)
                buffer_small = centroid.buffer(r)
                buffer_big = centroid.buffer(r + sample_step)
                search_range = buffer_big.difference(buffer_small)  # 做差，得到搜索范围(Do the difference and get the search range)
                inner_points = its_points[its_points.intersects(search_range)]  # 获取搜索范围内的点(Get the points within the search range)
                inner_points.reset_index(inplace=True, drop=True)

                if len(inner_points) != 0:
                    np.random.seed(0)
                    random_indices = np.random.randint(0, len(inner_points), points_per_step)
                    random_inner_points = inner_points.loc[random_indices]
                    sample_list.append(random_inner_points)
                    print(len(sample_list))

                else:
                    print("not found")
                    break  # 搜索范围内无点则跳出循环(If there is no point in the search range, the loop will be jumped out.)
            # print(len(sample_list))
            uniform_points = pd.concat(sample_list, ignore_index=True)
            uniform_points.loc[len(uniform_points), 'geometry'] = centroid
            points_list.append(uniform_points)
            indices += [i] * len(uniform_points)

    points_gdf = pd.concat(points_list, ignore_index=True)
    points_gdf['index'] = indices
    if points_gdf.crs is None:
        points_gdf.set_crs(merged_gdf.crs, inplace=True)
    return points_gdf


def create_prompt(approximate_building_plg, mode="centroid", point_fp=None, save=True, save_fp=None, sample_method=None):
    """
    基于指定方法构建提示(Build prompts based on the specified method)
    :param approximate_building_plg: 建筑物近似面数据(approximate building surface data)
    :param mode: 构建点提示策略，目前实现点提示的质心模式“centroid”、格网点模式"grid"、采样模式“sample”以及框提示的边界框模式"bound"(build point prompt strategy,
    currently implements the centroid mode "centroid", grid point mode "grid", sampling mode "sample" and bounding box mode "bound" for point prompts)
    :param save: 是否保存提示文件，默认为否(whether to save the prompt file, the default is no)
    :param save_fp: 保存路径，默认为None，若需保存文件则需要指定此参数(whether to save the prompt file, the default is no)
    :param sample_method: 采样方式, 默认为(sampling method, the default is yes)
    :return: 存放提示的gdf(gdf to store the prompts)
    """

    appro_plg_gdf = gpd.read_file(approximate_building_plg)
    if not isinstance(mode, str):
        mode = str(mode)
    func_name = "create_prompt_via_" + mode
    creation_func = globals().get(func_name)

    if not creation_func:
        print(r"No corresponding function found, please check whether the mode parameter is centroid (centroid mode), "
              r"sample (sampling mode), grid (grid point) or bound (bounding box mode)!")
        return
    else:
        if mode == "sample":
            if point_fp is None:
                print("The sample mode requires a point path to be specified, and clustered points are generally used. Please check")
                return
            else:
                print(r"Reading dot files")
                point_gdf = gpd.read_file(point_fp)
                print(f"The sampling method is： {sample_method}")
                prompt_gdf = creation_func(appro_plg_gdf, point_gdf, sample_method)
        else:
            prompt_gdf = creation_func(appro_plg_gdf)

    # 如果save为True且指定了保存路径则保存文件并返回否则返回prompt_gdf，否则只返回prompt_gdf
    # If save is True and the save path is specified, save the file and return otherwise return prompt_gdf, otherwise just return prompt_gdf
    if save:
        if save_fp is not None:
            prompt_gdf.to_file(save_fp)
            return prompt_gdf
        else:
            print("The file save path is empty and the file save fails.")
            return prompt_gdf
    else:
        return prompt_gdf


def grid_filter(points_fp, resolution, gap, result_fp=None, fnet_fp=None, its_fp=None):
    points = gpd.read_file(points_fp)
    sindex = points.sindex

    minx = np.min(points.X)
    maxx = np.max(points.X)
    miny = np.min(points.Y)
    maxy = np.max(points.Y)

    num_x = int((maxx - minx) / resolution) + 1
    num_y = int((maxy - miny) / resolution) + 1

    fnet_polygons = []
    its_polygons = []
    its_plg_zs = []
    for i in range(num_x):
        print(f"Fishnet calculation in progress，i：{i}")
        for j in range(num_y):
            left = minx + i * resolution
            bottom = miny + j * resolution
            right = left + resolution
            top = bottom + resolution
            polygon = Polygon([(left, bottom), (left, top), (right, top), (right, bottom), (left, bottom)])

            possible_matches_index = sindex.intersection(polygon.bounds)
            possible_matches = points.iloc[possible_matches_index]
            its_points = possible_matches[possible_matches.intersects(polygon)]

            # 如果该多边形内有建筑物点分布则计算其平均高度
            # If there are building points distributed within the polygon, calculate its average height
            if len(its_points) != 0:
                its_polygons.append(polygon)
                its_plg_zs.append(np.mean(its_points['Z']))

            # 如果保存渔网文件
            # If you save the fishnet file
            if fnet_fp is not None:
                fnet_polygons.append(polygon)

    # 筛选
    # Filter
    its_df = pd.DataFrame(data=np.array(its_plg_zs).T, columns=['Zmean'])
    its_gdf = gpd.GeoDataFrame(data=its_df, geometry=its_polygons, crs=points.crs)

    min_zmean = min(its_plg_zs)
    max_zmean = max(its_plg_zs)

    z_slice_fnets = []
    # 简化处理
    # Simplify processing
    for i in range(int(min_zmean), int(max_zmean), gap):
        print(f"Elevation slice calculation in progress，i：{i}")
        z_slice = its_gdf[(its_gdf['Zmean'] > i) & (its_gdf['Zmean'] < (i + gap))]
        if len(z_slice) != 0:
            z_slice_fnets.append(z_slice)

    z_slice_plgs = []
    for i in z_slice_fnets:
        print(f"Data fusion in progress")
        # 融合
        # Fusion
        z_slice_plg = i.dissolve().explode(ignore_index=True)
        z_slice_plgs.append(z_slice_plg)

    result_plg = pd.concat(z_slice_plgs)
    result_plg = result_plg[result_plg.area > 100]  # 过滤（Filter）
    result_plg.reset_index(inplace=True, drop=True)

    # 对点云再次分类
    # Reclassify the point cloud
    out_points = points.copy()
    out_points['cls'] = 999  # 初始化（initialization）
    for i in range(len(result_plg)):
        print(f"Point intersection operation in progress，i：{i}")
        plg_geom = result_plg.loc[i, 'geometry']
        possible_matches_index = sindex.intersection(plg_geom.bounds)
        possible_matches = points.iloc[possible_matches_index]
        its_points = possible_matches[possible_matches.intersects(plg_geom)]

        out_points.loc[its_points.index, 'cls'] = i

    out_points = out_points[out_points['cls'] != 999]

    if result_fp is not None:
        out_points.to_file(result_fp)
        print(f"The result data is saved： {result_fp}")

    if fnet_fp is not None:
        fnet_gdf = gpd.GeoDataFrame(geometry=fnet_polygons, crs=points.crs)
        fnet_gdf.to_file(fnet_fp)
        print(f"Fishing net data saved： {fnet_gdf}")

    if its_fp is not None:
        its_gdf.to_file(its_fp)
        print(f"Point intersection fishnet data saved： {fnet_gdf}")


def process_shp_files(shp1_path, shp2_path, output_shp_path):
    # 读取两个shp文件
    # Read two shp files
    shp1 = gpd.read_file(shp1_path)
    shp2 = gpd.read_file(shp2_path)

    # 创建一个空的GeoDataFrame来存储结果
    # Create an empty GeoDataFrame to store the results
    results = gpd.GeoDataFrame(columns=['geometry'])

    for idx1, geom1 in shp1.iterrows():
        intersecting_geoms = shp2[shp2.intersects(geom1['geometry'])]

        # 过滤出相交面积大于60%的几何体
        # Filter out geometries with an intersection area greater than 60%
        large_intersection_geoms = []
        for idx2, geom2 in intersecting_geoms.iterrows():
            intersection_area = geom1['geometry'].intersection(geom2['geometry']).area
            union_area = geom1['geometry'].union(geom2['geometry']).area
            if intersection_area / union_area > 0.6:
                large_intersection_geoms.append(geom2)

                # 根据相交几何体的数量进行处理
                # Process according to the number of intersecting geometries
        if len(large_intersection_geoms) > 1:
            print(f"Error: Multiple intersections found for geometry {idx1} in the first shapefile.")
        elif len(large_intersection_geoms) == 1:
            # 只有一个相交几何体，计算最大x、y和最小x、y来创建新面
            # There is only one intersecting geometry, calculate the maximum x, y and minimum x, y to create a new face
            geom2 = large_intersection_geoms[0]

            bounds1 = geom1['geometry'].bounds
            bounds2 = geom2['geometry'].bounds

            minx = min(bounds1[0], bounds2[0])
            miny = min(bounds1[1], bounds2[1])
            maxx = max(bounds1[2], bounds2[2])
            maxy = max(bounds1[3], bounds2[3])

            new_geom = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
            results = results._append({'geometry': new_geom}, ignore_index=True)

            # 设置输出的坐标系与输入的shp1相同
            # Set the output coordinate system to be the same as the input shp1
    results.crs = shp1.crs
    results['index'] = [i for i in range(len(results))]
    # 保存结果到新的shp文件
    # Save the results to a new shp file
    results.to_file(output_shp_path, driver='ESRI Shapefile')


def filter_and_update_build_idx(shp_file_path, output_shp_file_path):
    """
    读取SHP文件，去掉面积为0或小于所有面平均面积十分之一的面，并更新大于被去掉面build_idx的面的build_idx值。（Read the SHP file, remove the faces with an area
    of 0 or less than one tenth of the average area of all faces,and update the build_idx value of the faces larger than the build_idx of the
    removed faces.）
    :param shp_file_path: 输入的SHP文件路径（Input SHP file path）
    :param output_shp_file_path: 输出的SHP文件路径（Output SHP file path）
    :return: None
    """
    # 读取SHP文件
    # Read SHP file
    gdf = gpd.read_file(shp_file_path)

    # 如果gdf为空，则无需进行后续处理
    # If gdf is empty, no further processing is required
    if gdf.empty:
        print("The input shapefile is empty. No processing required.")
        return

        # 计算所有面的面积平均值
    # Calculate the average area of all faces
    average_area = gdf['geometry'].area.mean()

    # 找出面积为0或小于平均面积十分之一的面
    # Find faces whose area is 0 or less than one tenth of the average area
    small_areas_mask = (gdf['geometry'].area == 0) | (gdf['geometry'].area < average_area / 20)
    small_areas_gdf = gdf[small_areas_mask]

    # 获取需要去掉的面的build_idx值
    # Get the build_idx value of the face to be removed
    indices_to_remove = small_areas_gdf['build_idx'].tolist()

    # 对build_idx进行更新
    # Update build_idx
    for idx in indices_to_remove:
        mask = gdf['build_idx'] > idx
        gdf.loc[mask, 'build_idx'] -= 1

        # 过滤掉不需要的面
        # Filter out unnecessary faces
    filtered_gdf = gdf[~small_areas_mask]

    # 保存结果到新的SHP文件
    # Save the results to a new SHP file
    filtered_gdf.to_file(output_shp_file_path, driver='ESRI Shapefile')


def filter_shapefile(input_shp, output_shp):
    # 读取SHP文件
    # Read SHP file
    gdf = gpd.read_file(input_shp)

    # 计算每个面的面积
    # Calculate the area of ​​each face
    gdf['area'] = gdf.geometry.area

    # 计算平均面积
    # Calculate the average area
    average_area = gdf['area'].mean()

    # 找出需要删除的面（面积为0或小于平均面积十分之一）
    # Find the faces that need to be deleted (area is 0 or less than one tenth of the average area)
    to_remove = (gdf['area'] == 0) | (gdf['area'] < average_area / 30)

    # 更新大于要删除面Index的其他面的Index
    # Update the index of other faces that are greater than the index of the face to be deleted
    for index, row in gdf[to_remove].iterrows():
        gdf.loc[gdf['index'] > row['index'], 'index'] -= 1

        # 删除不符合条件的面
        # Delete faces that do not meet the criteria
    gdf_filtered = gdf[~to_remove]

    # 删除临时的面积列并保存结果
    # Delete the temporary area column and save the result
    gdf_filtered.drop(columns=['area']).to_file(output_shp, driver='ESRI Shapefile')


def merge_bbox(bbox1_fp, bbox2_fp, save_fp=None, build_idx=True):
    """
    基于两个bounding box矢量数据 计算他们的共同bounding box(Calculate the common bounding box based on two bounding box vector data)
    :param bbox1_fp:第一个框提示的保存路径（The save path of the first bounding box）
    :param bbox2_fp:第二个框提示的保存路径（The save path of the second bounding box）
    :param save_fp:结果保存路径（Result save path）
    :param build_idx:建筑物编号（Building Number）
    :return:
    """
    bbox1 = gpd.read_file(bbox1_fp)
    bbox2 = gpd.read_file(bbox2_fp)

    bounds1 = bbox1.bounds
    bounds2 = bbox2.bounds

    new_polygons = []
    for idx1 in bbox1.index:
        geom1 = bbox1.loc[idx1, 'geometry']
        bbox2_its1 = bbox2[bbox2.intersects(geom1)]
        its_idx = []
        for idx2 in bbox2_its1.index:
            geom2_its1 = bbox2_its1.loc[idx2, 'geometry']
            min_geom_area = min(geom1.area, geom2_its1.area)
            area_its = geom2_its1.intersection(geom1).area
            if area_its > (min_geom_area * 0.6):
                its_idx.append(idx2)

        zmin_x=0
        zmin_y = 0
        zmax_x = 0
        zmax_y = 0
        if len(its_idx) > 1:
            # print(f"bbox1中编号为{idx1}的bbox与bbox2中多个bbox对应，请检查")
            # return
            zmin_x = bounds2.loc[its_idx, 'minx'].min()
            zmin_y = bounds2.loc[its_idx, 'miny'].min()
            zmax_x = bounds2.loc[its_idx, 'maxx'].max()
            zmax_y = bounds2.loc[its_idx, 'maxy'].max()

            new_minx = min(bounds1.loc[idx1, 'minx'],zmin_x)
            new_miny = min(bounds1.loc[idx1, 'miny'], zmin_y)
            new_maxx = max(bounds1.loc[idx1, 'maxx'],zmax_x)
            new_maxy = max(bounds1.loc[idx1, 'maxy'],zmax_y)
            new_bbox = box(new_minx, new_miny, new_maxx, new_maxy)
            new_polygons.append(new_bbox)
        if len(its_idx) ==0:
            print(idx1)
            return
        if len(its_idx) == 1:

         new_minx = min(bounds1.loc[idx1, 'minx'], bounds2.loc[its_idx[0], 'minx'])
         new_miny = min(bounds1.loc[idx1, 'miny'], bounds2.loc[its_idx[0], 'miny'])
         new_maxx = max(bounds1.loc[idx1, 'maxx'], bounds2.loc[its_idx[0], 'maxx'])
         new_maxy = max(bounds1.loc[idx1, 'maxy'], bounds2.loc[its_idx[0], 'maxy'])
         new_bbox = box(new_minx, new_miny, new_maxx, new_maxy)
         new_polygons.append(new_bbox)

    if save_fp is not None:
        gdf = gpd.GeoDataFrame(geometry=new_polygons, crs=bbox1.crs)

        if build_idx:
            gdf['index'] = [i for i in range(len(gdf))]

        gdf.to_file(save_fp)


def cal_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity


def save_mask2tif(binary_mask, tif_save_fp, meta):
    out_arr = (binary_mask * 255).astype(np.uint8)
    with rio.open(tif_save_fp, "w", **meta) as dst:
        for i in range(meta["count"]):
            dst.write(out_arr, i + 1)
    print(f"The raster is saved：{tif_save_fp}")


def save_mask2shp(binary_mask, shp_save_fp, meta, build_idx=False):
    simu_arr = np.zeros((meta['height'], meta['width']), dtype=np.uint8)
    shapes = features.shapes(simu_arr, mask=binary_mask, transform=meta['transform'])
    fc = [{"geometry": shapely.geometry.shape(shape), "properties": None} for shape, value in shapes]
    out_gdf = gpd.GeoDataFrame.from_features(fc)
    out_gdf.set_crs(crs=meta['crs'], inplace=True)
    if build_idx:
        out_gdf['build_idx'] = [i for i in range(len(out_gdf))]
    out_gdf.to_file(shp_save_fp)
    print(f"The vector is saved：{shp_save_fp}")


def extract_fea_via_sam(img_fp, ckpt_path, save_fp=None, sam_model_type='vit_h', device='cuda', norm=False):
    """
    利用sam提取图像特征(Extract image features using sam)
    :param img_fp: 图像路径(image path)
    :param save_fp: 特征保存路径，后缀为.npy(feature save path, suffix is .npy)
    :param ckpt_path: SAM模型权重路径(SAM model weight path)
    :param sam_model_type: sam模型模式，默认为"vit_h"(sam model mode, default is "vit_h")
    :param device: 计算设备，默认为"cuda"(computing device, default is "cuda")
    :param norm: 是否对特征进行归一化，默认为否(whether to normalize the features, default is no)
    :return:
    """
    feature_extractor = FeatureExtractorV2(checkpoint=ckpt_path,
                                           sam_model_type=sam_model_type,
                                           device=device)
    fea_32d = feature_extractor.extract_feature(img_fp, is_norm=norm)

    if save_fp is not None:
        np.save(file=save_fp, arr=fea_32d)

    return fea_32d


def renew_appr_plg_via_fea32d(tif_fp,
                              ch_fp,
                              fea_fp,
                              n_seg,
                              similarity_threshold=0.999,
                              tif_save_fp=None,
                              shp_save_fp=None):
    """
    近似面优化(Approximate surface optimization)
    :param tif_fp: 地块影像(plot image)
    :param ch_fp: 凸包多边形矢量文件路径(convex hull polygon vector file path)
    :param fea_fp: SAM特征路径(SAM feature path)
    :param n_seg: 超像素分割个数(number of superpixel segmentation)
    :param similarity_threshold: 特征相似度阈值(feature similarity threshold)
    :param tif_save_fp: 栅格保存路径(raster save path)
    :param shp_save_fp: 矢量保存路径(vector save path)
    :return:
    """
    src = rio.open(tif_fp)
    if src.meta['nodata'] is not None:
        nodata =src.meta['nodata']
    else:
        nodata=0
    feature = np.load(fea_fp) if isinstance(fea_fp, str) else fea_fp
    ch_gdf = gpd.read_file(ch_fp)

    rgb_img = np.transpose(src.read(), (1, 2, 0))[:, :, :3]
    non_zero_mask = ~np.all(rgb_img == nodata, axis=2)

    if n_seg==1:
        n_seg =round(non_zero_mask.sum()/(1024*1024))*500

    print(n_seg)

    segments = slic(rgb_img, n_segments=n_seg, compactness=10, sigma=3, start_label=1, mask=non_zero_mask).astype(np.uint16)

    renew_segments = segments.copy()  #
    building_labels = []
    for i in ch_gdf.index:
        new_label = n_seg + i + 1
        geom = ch_gdf.loc[i, "geometry"]
        geom_json = [geom.__geo_interface__]
        masked_img, _ = mask(src, geom_json)
        overlap_spx_mask = np.all(masked_img != nodata, axis=0)
        overlap_spx_values = renew_segments[overlap_spx_mask]
        overlap_spx_uni = np.unique(overlap_spx_values)
        for u in overlap_spx_uni:
            spx_area = len(renew_segments[renew_segments == u])
            overlap_area = len(overlap_spx_values[overlap_spx_values == u])
            if (overlap_area / spx_area) > 0.5:
                building_labels.append(new_label)
                renew_segments[renew_segments == u] = new_label
        renew_segments[overlap_spx_mask] = new_label

    graph = RAG(renew_segments)

    renew_s2 = []
    for b_idx in building_labels:
        # print(b_idx)
        ndfea = feature.shape[2]
        seeds = [b_idx]
        marks = []
        while len(seeds) > 0:
            seed = seeds.pop()
            seed_mask = renew_segments == seed
            fea_vector_seed = np.stack([np.mean(feature[:, :, i][seed_mask]) for i in range(ndfea)], axis=-1)
            neighbor_list = list(graph.neighbors(seed))
            neighbor_list = [i for i in neighbor_list if i != 0]

            for neighbor in neighbor_list:

                if neighbor < n_seg and neighbor not in marks:
                    neighbor_mask = renew_segments == neighbor
                    fea_vector_neighbor = np.stack([np.mean(feature[:, :, i][neighbor_mask]) for i in range(ndfea)], axis=-1)
                    cos_sim = cal_cosine_similarity(fea_vector_seed, fea_vector_neighbor)

                    if cos_sim > similarity_threshold:
                        marks.append(neighbor)
                        renew_s2.append([b_idx, neighbor])
                        seeds.append(neighbor)

    for renew_couple in renew_s2:
        renew_segments[renew_segments == renew_couple[1]] = renew_couple[0]

    renew_segments[renew_segments <= (n_seg)] = 0
    renew_segments[renew_segments > n_seg] = 1
    renew_segments = renew_segments.astype(bool)

    if tif_save_fp is not None:
        save_mask2tif(renew_segments, tif_save_fp, src.meta)

    if shp_save_fp is not None:
        save_mask2shp(renew_segments, shp_save_fp, src.meta, build_idx=True)


def las_classification(las_inputpath, las_outputpath):
    """
    由于使用ENVI—Lidar对点云进行分类后ENVI只会对不同类别的点用字段值进行区分并不会为不同的点新建一个点云文件，所以可以用这个方法为不同类别的点单独建立一个文件
    (Since ENVI-Lidar only distinguishes different types of points with field values after classifying the point cloud and does not
    create a new point cloud file for different points, this method can be used to create a separate file for different types of points.)
    :param las_inputpath: ENVI分类好后的点云文件(Point cloud file after ENVI classification)
    :param las_outputpath: 保存路径(save path)
    """
    las = laspy.read(las_inputpath)
    label = las.classification

    value_cnt = {}
    for value in label:
        value_cnt[value] = value_cnt.get(value, 0) + 1

    for j in value_cnt.keys():

        new_file = laspy.create(point_format=las.header.point_format, file_version=las.header.version)

        result = label == j
        new_file.points = las.points[result]

        lasoutpath = f"{las_outputpath}_{j}.las"

        new_file.write(lasoutpath)

        print(f"finish {lasoutpath}")


def filter_points_below_threshold(DEM_path, las_inputpath,las_outputpath,height_threshold):
    """
    将低于某一高度阈值的点进行保留（Points below a certain height threshold are retained）
    :param DEM_path: 保存路径(DEM path)
    :param las_inputpath: 点云文件路径(Point cloud file)
    :param las_outputpath: 保存路径(save path)
    :param height_threshold: 高度阈值(height threshold)
    """
    ds = rio.open(DEM_path)
    transform = ds.transform
    band1 = ds.read(1)

    laspath = las_inputpath
    las = laspy.read(laspath)
    las_header = las.header
    new_file_c0 = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_file_c0.x = las.x
    new_file_c0.y = las.y
    new_file_c0.z = las.z

    for i in range(las_header.point_records_count):
        row, col = rio.transform.rowcol(transform, las.x[i], las.y[i])
        height = las.z[i] - band1[row, col]
        if 0 > height or 1000 < height:
            new_file_c0.z[i] = 1
        else:
            new_file_c0.z[i] = height

    new_file_c0.intensity = las.intensity

    def multiple_conditions(value1):
        condition1 = value1 < height_threshold
        return condition1

    result = np.vectorize(multiple_conditions)(new_file_c0.z)
    new_file_c0.points = new_file_c0.points[result]
    outpath=las_outputpath
    new_file_c0.write(outpath)
    print("finish",outpath)


def filter_points_above_threshold(DEM_path, las_inputpath,las_outputpath,height_threshold):
    """
    将低于某一高度阈值的点进行保留（Points above a certain height threshold are retained）
    :param DEM_path: 保存路径(DEM path)
    :param las_inputpath: 点云文件路径(Point cloud file)
    :param las_outputpath: 保存路径(save path)
    :param height_threshold: 高度阈值(height threshold)
    """
    ds = rio.open(DEM_path)
    transform = ds.transform
    band1 = ds.read(1)

    laspath = las_inputpath
    las = laspy.read(laspath)
    las_header = las.header
    new_file_c0 = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_file_c0.x = las.x
    new_file_c0.y = las.y
    new_file_c0.z = las.z

    for i in range(las_header.point_records_count):
        row, col = rio.transform.rowcol(transform, las.x[i], las.y[i])
        height = las.z[i] - band1[row, col]
        if 0 > height or 1000 < height:
            new_file_c0.z[i] = 1
        else:
            new_file_c0.z[i] = height

    new_file_c0.intensity = las.intensity

    def multiple_conditions(value1):
        condition1 = value1 > height_threshold
        return condition1

    result = np.vectorize(multiple_conditions)(new_file_c0.z)
    new_file_c0.points = new_file_c0.points[result]
    outpath=las_outputpath
    new_file_c0.write(outpath)
    print("finish",outpath)

