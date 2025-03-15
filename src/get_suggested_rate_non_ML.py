def lookup_crf(avg_motion, avg_edge_density, avg_texture,
               motion_max=5.0, edge_max=0.5, texture_max=10.0,
               crf_static=40, crf_dynamic=18):
    """
    Map the average scene features to a CRF value using weighted linear interpolation.
    
    Parameters:
      avg_motion: Average motion metric from the scene.
      avg_edge_density: Average edge density (fraction between 0 and 1).
      avg_texture: Average texture complexity (e.g., entropy).
      motion_max, edge_max, texture_max:
          Estimated maximum values for normalization.
      crf_static: CRF value for a very static/low-complexity scene.
      crf_dynamic: CRF value for a highly dynamic/complex scene.
    
    Returns:
      An integer CRF value interpolated between crf_static and crf_dynamic.
    """
    # Normalize each feature to a range of 0 to 1.
    norm_motion = min(avg_motion / motion_max, 1.0)
    norm_edge = min(avg_edge_density / edge_max, 1.0)
    norm_texture = min(avg_texture / texture_max, 1.0)
    
    # Create a weighted composite score.
    # Here we assume motion is most important, then edge density, then texture.
    weight_motion = 0.5
    weight_edge = 0.3
    weight_texture =0.2
    composite_score = (weight_motion * norm_motion +
                       weight_edge * norm_edge +
                       weight_texture * norm_texture)
    
    # Use linear interpolation:
    # composite_score=0 -> CRF = crf_static (more compression)
    # composite_score=1 -> CRF = crf_dynamic (better quality)
    crf = crf_static - composite_score * (crf_static - crf_dynamic)
  
    #print("Normalized Metrics:")
    #print("norm_motion =", norm_motion)
    #print("norm_edge =", norm_edge)
    #print("norm_texture =", norm_texture)
    print("Composite Score =", composite_score)
    return round(crf)
    