import src.utils
import src.building_seg_via_SAM
import convert_tools.convert_tools
import geopandas as gpd

# Process the classification results of ENVI-Lidar to obtain the building category point cloud
src.utils.las_classification("pointCloud_000.las","save_path")

# Perform elevation filtering on the building category point cloud to remove low buildings, etc.
src.utils.filter_points_below_threshold("DEM.tif","las_file","save_path","threshold")

# Convert the result of the previous step to a different format
convert_tools.convert_tools.las2shp("las_path","shp_path")

# Clustering point clouds
src.utils.cluster2d("shp_path","save_path")

# Constructing the convex hull
src.utils.cal_convex_hull("cluster_path","save_path")

# Extracting image features using SAM
src.utils.extract_fea_via_sam("Demo.tif","model/sam_vit_h_4b8939.pth","save_path")

# The convex hull of the building constructed by the point cloud is improved by superpixels
# and the SAM image features extracted in the previous step.
# Different superpixel parameters can be set to obtain different results
src.utils.renew_appr_plg_via_fea32d("Demo.tif","convex_hull_path","fea_via_sam_path",1000,0.99999,"save_path")
src.utils.renew_appr_plg_via_fea32d("Demo.tif","convex_hull_path","fea_via_sam_path",1500,0.99999,"save_path")

# Build Prompts
# If you want to build a prompt other than the BBox, you need to add a "build_idx" field
# to the input building approximate surface data, and the field starts counting from 0.
# 1、BBOX
gdf11 = gpd.read_file("renew_appr_plg_via_fea32d_1000_path")
gdf22 = gpd.read_file("renew_appr_plg_via_fea32d_1500_path")
points_gdf = src.utils.create_prompt_via_bound(gdf11)
points_gdf.to_file("save_path1", driver='ESRI Shapefile')
points_gdf1 = src.utils.create_prompt_via_bound(gdf22)
points_gdf1.to_file("save_path2", driver='ESRI Shapefile')
src.utils.filter_shapefile("save_path1","save_path1")
src.utils.filter_shapefile("save_path2","save_path2")
src.utils.merge_bbox("save_path1","save_path2","save_path3")
# 2、centroid
src.utils.create_prompt("renew_appr_plg_via_fea32d_1000_path", "centroid","",True,"save_path")
# 3、uniform
gdf1 = gpd.read_file("renew_appr_plg_via_fea32d_1000_path")
gdf2 = gpd.read_file("cluster_path")
points_gdf = src.utils.create_prompt_via_sample(gdf1,gdf2,"uniform")
points_gdf.to_file("save_path")
# 4、grid
src.utils.create_prompt("renew_appr_plg_via_fea32d_1000_path", "grid","",True,"save_path")
# 5、random
gdf1 = gpd.read_file("renew_appr_plg_via_fea32d_1000_path")
gdf2 = gpd.read_file("cluster_path")
points_gdf = src.utils.create_prompt_via_sample(gdf1,gdf2,"centroid_random")
points_gdf.to_file("save_path")

# SAM segmentation using box Prompts
src.building_seg_via_SAM.sam_seg_single("Demo.tif","Prompt_path",
                                        "renew_appr_plg_via_fea32d_1000_path","save_path1",
                                        "save_path2","model/sam_vit_h_4b8939.pth")

