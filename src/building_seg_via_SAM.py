import rasterio as rio
from rasterio import features
import geopandas as gpd
import cv2
import shapely
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from skimage.morphology import closing, square
from skimage.measure import label
from shapely import Polygon


def post_process(seg_mask, save_fp=None, kernel_size=20, pixel_threshold=300):
    """
    分割结果后处理（Post-processing of segmentation results）
    :param seg_mask: 分割结果，文件路径或数组（segmentation results, file path or array）
    :param save_fp: 保存路径（save path）
    :param kernel_size: 侵蚀核大小（erosion kernel size）
    :param pixel_threshold: 像素阈值（pixel threshold）
    :return:
    """
    if isinstance(seg_mask, str):
        with rio.open(seg_mask) as src:
            result_arr = src.read(1)
            meta = src.meta
    else:
        result_arr = seg_mask * 255

    closed_arr = closing(result_arr, square(kernel_size))
    labeled_closed_arr = label(closed_arr, background=0)

    for i in np.unique(labeled_closed_arr):
        if i != 0:  # 背景值会被标记为0
            label_mask = labeled_closed_arr == i
            if label_mask.sum() < pixel_threshold:
                closed_arr[label_mask] = 0

    if save_fp is not None:
        with rio.open(save_fp, "w", **meta) as dst:
            for i in range(meta["count"]):
                dst.write(closed_arr, i + 1)

    closed_arr[closed_arr > 0] = 1
    closed_arr = closed_arr.astype(bool)

    return closed_arr


def generate_obb(seg_result_fp, obb_save_fp):
    """
    基于分割结果生成建筑物有向边界框（Generate a building oriented bounding box based on the segmentation result）
    :param seg_result_fp: 分割结果路径，tif格式（segmentation result path, tif format）
    :param obb_save_fp: 有向边界框保存路径（oriented bounding box save path）
    :return:
    """
    with rio.open(seg_result_fp) as src:
        result_mask = src.read(1)
        transform = src.transform
        crs = src.crs

    labeled_arr = label(result_mask, background=0)

    geo_obbs = []
    for i in np.unique(labeled_arr).tolist():
        if i != 0:
            rows, cols = np.where(labeled_arr == i)
            inner_points = np.stack([cols, rows], axis=-1)
            rect = cv2.minAreaRect(inner_points)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            box_geo_coors = [transform * i for i in box]
            geo_box = Polygon(box_geo_coors)
            geo_obbs.append(geo_box)

    obb_gdf = gpd.GeoDataFrame(geometry=geo_obbs, crs=crs)
    obb_gdf.to_file(obb_save_fp)


def sam_seg_single(tif_fp, prompt,
                   appr_plg_fp,
                   tif_save_fp=None,
                   shp_save_fp=None,
                   ckpt_path=None,
                   model_type='vit_h',
                   device='cuda'):
    """
    利用SAM基于单类型提示进行分割，支持自动判定为点提示或框提示（Use SAM to segment based on a single type of prompt,
    support automatic determination of point prompt or box prompt）
    :param tif_fp: 待分割图像路径（image path to be segmented）
    :param prompt: 存放提示的shp文件路径（shp file path for storing prompts）
    :param appr_plg_fp: 存放建筑物近似面数据的shp文件路径（shp file path for storing approximate building surface data）
    :param tif_save_fp: tif格式结果保存路径（tif format result save path）
    :param shp_save_fp: shp格式结果保存路径（shp format result save path）
    :param ckpt_path: 权重路径，建议显示指定，否则将自动下载权重（weight path, it is recommended to specify it explicitly,
    otherwise the weight will be automatically downloaded）
    :param model_type: SAM模型类别（SAM model category）
    :param device: 用于计算的设备（device used for calculation）
    :return:
    """
    ### 获取近似建筑物矢量面 ###
    ### Get approximate building vector faces ###
    appr_gdf = gpd.read_file(appr_plg_fp)
    idx_uni = appr_gdf['build_idx'].to_list()  # idx_uni: 建筑物实例编号(Building Instance Number)

    ### 获取影像信息 ###
    ### Get image information ###
    with rio.open(tif_fp) as src:
        transform = src.transform
        meta = src.meta
        data_shape = src.read(1).shape
        crs = src.crs

        if src.count == 3:
            img = np.transpose(src.read(), (1, 2, 0))  # (C, H, W) -> (H, W ,C)
        elif src.count > 3:
            img = np.stack([src.read(1), src.read(2), src.read(3)], axis=2)  # stack 3*(H, W) -> (H, W, 3)
        else:
            print("The input image band is less than 3, please check")

    ### 加载模型 ###
    ### Loading the model ###
    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("sam initialized")

    ### 获取图像嵌入 ###
    ### Get image embed ###
    predictor.set_image(img)
    print("image embedding gotten")

    ### 获取提示数据 ###
    ### Get prompt data ###
    prompt_gdf = gpd.read_file(prompt) if isinstance(prompt, str) else prompt  # 如果prompt是文件路径则打开文件
                                                                               # (If prompt is a file path, open the file)

    ### 基于提示数据利用SAM分割影像 ###
    ### Use SAM to segment the image based on the prompt data ###
    # 检查prompt_gdf的矢量类型
    # Check prompt_gdf's vector type
    if len(prompt_gdf.geom_type.unique()) != 1:  # 只有一种类型(Only one type)
        print("The prompt contains multiple types of vectors, please check")
        return

    else:
        seg_results = []
        if prompt_gdf.geom_type.unique()[0] == 'Point':
            for idx in idx_uni:
                prompt_gdf_idx = prompt_gdf[prompt_gdf['index'] == idx]
                prompt_gdf_idx.reset_index(inplace=True, drop=True)
                prompt_gs_idx = prompt_gdf_idx.geometry

                x_list = prompt_gs_idx.x.tolist()
                y_list = prompt_gs_idx.y.tolist()

                pixel_ys, pixel_xs = rio.transform.rowcol(transform, x_list, y_list)
                prompt_idx = np.array([list(a) for a in zip(pixel_xs, pixel_ys)])
                prompt_label_idx = np.array(([1] * len(prompt_idx)))

                masks, _, _ = predictor.predict(point_coords=prompt_idx,
                                                point_labels=prompt_label_idx,
                                                multimask_output=True)  # 指定同时输出多个分割结果
                                                                        # (Specify to output multiple segmentation results simultaneously)

                # 比较面积, batch_transform[0]: 影像空间分辨率，要求影像在投影坐标系下
                # Compare areas, batch_transform[0]: image spatial resolution, requiring images to be in a projected coordinate system
                dst_area = appr_gdf.loc[idx, 'geometry'].area  # 目标面积(Target area)
                mask_areas = [masks[i].sum() * transform[0] * transform[0] for i in range(3)]  # 结果面积列表(Result area list)
                best_mask_index = mask_areas.index(min(mask_areas, key=lambda x: abs(x - dst_area)))  # 获取面积最接近的mask的序号
                                                                                                      # (Get the number of the mask with the closest area)
                seg_results.append(masks[best_mask_index])

        elif prompt_gdf.geom_type.unique()[0] == 'Polygon':  # 边界框提示(Bounding Box Hint)
            for idx in idx_uni:
                prompt_gdf_idx = prompt_gdf[prompt_gdf['index'] == idx]
                prompt_gdf_idx.reset_index(inplace=True, drop=True)
                prompt_gs_idx = prompt_gdf_idx.geometry

                if len(prompt_gs_idx) != 1:  # 只能由一个边界框提示(Can only be hinted by a bounding box)
                    print("One approximate face corresponds to multiple bounding boxes, please check")
                    return

                else:
                    bound_idx = prompt_gs_idx.bounds.loc[0]
                    pixel_miny, pixel_minx = rio.transform.rowcol(transform, bound_idx['minx'], bound_idx['maxy'])
                    pixel_maxy, pixel_maxx = rio.transform.rowcol(transform, bound_idx['maxx'], bound_idx['miny'])
                    prompt_idx = np.array([pixel_minx, pixel_miny, pixel_maxx, pixel_maxy])  # 构建提示(Build Tips)
                    masks, _, _ = predictor.predict(box=prompt_idx,
                                                    multimask_output=True)

                    dst_area = appr_gdf.loc[idx, 'geometry'].area  # 目标面积(Target area)
                    mask_areas = [masks[i].sum() * transform[0] * transform[0] for i in range(masks.shape[0])]  # 结果面积列表(Result area list)
                    best_mask_index = mask_areas.index(min(mask_areas, key=lambda x: abs(x - dst_area)))  # 获取面积最接近的mask的序号
                                                                                                          # (Get the number of the mask with the closest area)
                    seg_results.append(masks[best_mask_index])
        else:
            print("Please check the vector data type")
            return

        result = np.any(seg_results, axis=0)  # 叠加所有分割结果(Overlay all segmentation results)
        result = post_process(result)  # 后处理(Post-processing)

        # 保存结果(Save the results)
        if tif_save_fp is not None:
            out_arr = (result * 255).astype(np.uint8)
            with rio.open(tif_save_fp, "w", **meta) as dst:
                for i in range(meta["count"]):
                    dst.write(out_arr, i + 1)
            print(f"The segmentation result raster is saved：{tif_save_fp}")

        if shp_save_fp is not None:
            simu_arr = np.zeros(data_shape, dtype=np.uint8)
            shapes = features.shapes(simu_arr, mask=result, transform=transform)
            fc = [{"geometry": shapely.geometry.shape(shape), "properties": None} for shape, value in shapes]  # fc: feature collection
            out_gdf = gpd.GeoDataFrame.from_features(fc)
            out_gdf.set_crs(crs=crs, inplace=True)
            out_gdf.to_file(shp_save_fp)
            print(f"The segmentation result vector is saved：{shp_save_fp}")


def sam_seg_multi(tif_fp,
                  point_prompt,
                  box_prompt,
                  appr_plg_fp,
                  tif_save_fp=None,
                  shp_save_fp=None,
                  ckpt_path=None,
                  model_type='vit_h',
                  device='cuda'):
    """
    利用SAM基于多类型（点与框）提示进行分割(Segmentation based on multi-type (point and box) prompts using SAM)
    :param tif_fp: 待分割图像路径(Path to the image to be segmented)
    :param point_prompt: 存放点提示的shp文件路径(Path to the shp file storing point prompts)
    :param box_prompt: 存放框提示的shp文件路径(Path to the shp file storing box prompts)
    :param appr_plg_fp: 存放建筑物近似面数据的shp文件路径(Path to the shp file storing approximate building surface data)
    :param tif_save_fp: tif格式结果保存路径(Path to save results in tif format)
    :param shp_save_fp: shp格式结果保存路径(Path to save results in shp format)
    :param ckpt_path: 权重路径，建议显示指定，否则将自动下载权重(Path to weights, it is recommended to specify explicitly,
     otherwise the weights will be automatically downloaded)
    :param model_type: SAM模型类别(SAM model category)
    :param device: 用于计算的设备(Device used for calculation)
    :return:
    """

    ### 获取近似建筑物矢量面 ###
    ### Get approximate building vector surface ###
    appr_gdf = gpd.read_file(appr_plg_fp)
    idx_uni = appr_gdf['build_idx'].to_list()  # idx_uni: 建筑物实例编号(Building Instance Number)

    ### 获取影像信息 ###
    ### Get image information ###
    with rio.open(tif_fp) as src:
        transform = src.transform
        meta = src.meta
        data_shape = src.read(1).shape
        crs = src.crs

        if src.count == 3:
            img = np.transpose(src.read(), (1, 2, 0))  # (C, H, W) -> (H, W ,C)
        elif src.count > 3:
            img = np.stack([src.read(1), src.read(2), src.read(3)], axis=2)  # stack 3*(H, W) -> (H, W, 3)
        else:
            print("The input image band is less than 3, please check")

    ### 加载模型 ###
    ### Loading the model ###
    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("sam initialized")

    ### 获取图像嵌入 ###
    ### Get image embed ###
    predictor.set_image(img)
    print("image embedding gotten")

    ### 获取提示数据 ###
    ### Get prompt data ###
    p_prompts = gpd.read_file(point_prompt) if isinstance(point_prompt, str) else point_prompt
    b_prompts = gpd.read_file(box_prompt) if isinstance(box_prompt, str) else box_prompt

    ### 基于提示数据利用SAM分割影像 ###
    ### Use SAM to segment the image based on the prompt data ###
    if len(p_prompts.geom_type.unique()) != 1:
        print("The point prompt contains multiple types of vectors, please check")
        return

    elif len(b_prompts.geom_type.unique()) != 1:
        print("The box prompt contains multiple types of vectors, please check")
        return

    else:
        seg_results = []
        for idx in idx_uni:
            # 构建点提示
            # Build point prompt
            p_prompt_gdf_idx = p_prompts[p_prompts['index'] == idx]
            p_prompt_gdf_idx.reset_index(inplace=True, drop=True)
            p_prompt_gs_idx = p_prompt_gdf_idx.geometry

            px_list = p_prompt_gs_idx.x.tolist()
            py_list = p_prompt_gs_idx.y.tolist()

            pixel_pys, pixel_pxs = rio.transform.rowcol(transform, px_list, py_list)
            pprompt_idx = np.array([list(coords) for coords in zip(pixel_pxs, pixel_pys)])
            pprompt_label_idx = np.array(([1] * len(pprompt_idx)))

            # 构建框提示
            # Build box prompt
            b_prompt_gdf_idx = b_prompts[b_prompts['index'] == idx]
            b_prompt_gdf_idx.reset_index(inplace=True, drop=True)
            b_prompt_gs_idx = b_prompt_gdf_idx.geometry

            if len(b_prompt_gs_idx) != 1:
                print(f"The approximate surface with sequence number {idx} corresponds to multiple bounding boxes, please check")
                return

            else:
                bound_idx = b_prompt_gs_idx.bounds.loc[0]
                pixel_bminy, pixel_bminx = rio.transform.rowcol(transform, bound_idx['minx'], bound_idx['maxy'])
                pixel_bmaxy, pixel_bmaxx = rio.transform.rowcol(transform, bound_idx['maxx'], bound_idx['miny'])
                bprompt_idx = np.array([pixel_bminx, pixel_bminy, pixel_bmaxy, pixel_bmaxx])

            masks, _, _ = predictor.predict(point_coords=pprompt_idx,
                                            point_labels=pprompt_label_idx,
                                            box=bprompt_idx,
                                            multimask_output=True)

            dst_area = appr_gdf.loc[idx, 'geometry'].area
            mask_areas = [masks[i].sum() * transform[0] * transform[0] for i in range(masks.shape[0])]
            best_mask_index = mask_areas.index(min(mask_areas, key=lambda x: abs(x - dst_area)))
            seg_results.append(masks[best_mask_index])

    result = np.any(seg_results, axis=0)
    result = post_process(result)

    if tif_save_fp is not None:
        out_arr = (result * 255).astype(np.uint8)
        with rio.open(tif_save_fp, "w", **meta) as dst:
            for i in range(meta["count"]):
                dst.write(out_arr, i + 1)
        print(f"The segmentation result raster is saved：{tif_save_fp}")

    if shp_save_fp is not None:
        simu_arr = np.zeros(data_shape, dtype=np.uint8)
        shapes = features.shapes(simu_arr, mask=result, transform=transform)
        fc = [{"geometry": shapely.geometry.shape(shape), "properties": None} for shape, value in shapes]
        out_gdf = gpd.GeoDataFrame.from_features(fc)
        out_gdf.set_crs(crs=crs, inplace=True)
        out_gdf.to_file(shp_save_fp)
        print(f"The segmentation result vector is saved：{shp_save_fp}")