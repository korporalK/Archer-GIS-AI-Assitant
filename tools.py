from langchain.tools import tool
import arcpy
import os
from typing import List, Dict, Any, Union, Optional
import json
import time
import aiohttp
import asyncio
import aiohttp
import aiofiles
from datetime import datetime
import requests
from urllib.parse import urlparse
import fiona
import rasterio
import traceback
import tarfile


arcpy.env.overwriteOutput = True


# Helper Functions
def _feature_class_exists(feature_class: str) -> bool:
    """Checks if a feature class exists."""
    try:
        return arcpy.Exists(feature_class)
    except Exception:
        return False

def _dataset_exists(dataset: str) -> bool:
    """Checks if any dataset (feature class, table, raster, etc.) exists."""
    try:
        return arcpy.Exists(dataset)
    except Exception:
        return False

def _is_valid_field_name(feature_class: str, field_name: str) -> bool:
    """Checks if a field name is valid for a given feature class."""
    if not _feature_class_exists(feature_class):
        return False
    try:
        fields = arcpy.ListFields(feature_class)
        return any(field.name == field_name for field in fields)
    except Exception:
        return False

def _validate_raster_exists(raster_path: str) -> bool:
    """Validates that a raster exists and is accessible.
    
    Args:
        raster_path: Path to the raster to validate
        
    Returns:
        bool: True if the raster exists and is accessible, False otherwise
    """
    try:
        if not arcpy.Exists(raster_path):
            return False
        # Try to create a raster object to ensure it's a valid raster
        raster = arcpy.Raster(raster_path)
        return True
    except Exception:
        return False

# Feature Class Tools
@tool
def buffer_features(input_features: str, output_features: str, buffer_distance: float, buffer_unit: str) -> str:
    """Creates buffer polygons around input features at a specified distance.
    
    This tool generates buffer polygons around input points, lines, or polygons. Buffers can be 
    used for proximity analysis, creating protection zones, or visual enhancement of features.
    The buffer distance determines how far from each feature the buffer extends.
    
    GIS Concepts:
    - Buffers create new polygon features that represent areas within a specified distance of input features
    - Positive buffer distances create larger polygons around the inputs
    - Buffer unit determines the measurement units (e.g., meters, feet)
    - Useful for answering questions like "What's within X distance of this feature?"

    Args:
        input_features: The path to the input feature class (points, lines, or polygons).
                       Must be an existing feature class in a workspace.
        output_features: The path where the buffered feature class will be saved.
                        Will be overwritten if it already exists.
        buffer_distance: The distance to buffer around each feature.
                        Must be a positive number.
        buffer_unit: The unit of measurement for the buffer distance.
                    Valid values: 'Meters', 'Feet', 'Kilometers', 'Miles', 'NauticalMiles', or 'Yards'.

    Returns:
        A message indicating success or failure with details about the operation.
        
    Example:
        >>> buffer_features("D:/data/cities.shp", "D:/output/cities_buffer.shp", 1000, "Meters")
        "Successfully buffered D:/data/cities.shp to D:/output/cities_buffer.shp."
        
    Notes:
        - The input feature class must exist
        - The output will be overwritten if it already exists
        - The spatial reference of the output will match the input
    """
    try:
        if not arcpy.Exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."
        if not isinstance(buffer_distance, (int, float)):
            return "Error: buffer_distance must be a number."
        if buffer_unit.lower() not in ['meters', 'feet', 'kilometers', 'miles', 'nauticalmiles', 'yards']:
            return "Error: Invalid buffer_unit. Must be 'Meters', 'Feet', 'Kilometers', 'Miles', 'NauticalMiles', or 'Yards'."

        arcpy.analysis.Buffer(input_features, output_features, f"{buffer_distance} {buffer_unit}")
        return f"Successfully buffered {input_features} to {output_features}."
    except arcpy.ExecuteError:
        return f"Error buffering features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def clip_features(input_features: str, clip_features: str, output_features: str) -> str:
    """Clips the input features using the clip features.

    Args:
        input_features: The path to the input feature class to be clipped.
        clip_features: The path to the feature class used to clip the input features.
        output_features: The path to the output (clipped) feature class.
    """
    try:
        if not _feature_class_exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."
        if not _feature_class_exists(clip_features):
            return f"Error: Clip feature class '{clip_features}' does not exist."
        arcpy.analysis.Clip(input_features, clip_features, output_features)
        return f"Successfully clipped '{input_features}' with '{clip_features}' to '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error clipping features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def dissolve_features(input_features: str, output_features: str, dissolve_field: str = "", 
                     statistics_fields: str = "", multi_part: str = "MULTI_PART") -> str:
    """Dissolves features based on specified attributes.

    Args:
        input_features: The path to the input feature class.
        output_features: The path to the output feature class.
        dissolve_field: The field(s) to dissolve on (optional, comma-separated).
        statistics_fields: Fields to calculate statistics on (optional, format: "field_name SUM;field_name MEAN").
        multi_part: Whether to create multi-part features ("MULTI_PART" or "SINGLE_PART").
    """
    try:
        if not _feature_class_exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."

        if multi_part.upper() not in ("MULTI_PART", "SINGLE_PART"):
            return "Error: Invalid value for 'multi_part'. Must be 'MULTI_PART' or 'SINGLE_PART'."

        arcpy.management.Dissolve(input_features, output_features, dissolve_field, statistics_fields, multi_part)
        return f"Successfully dissolved features from '{input_features}' to '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error dissolving features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def merge_features(input_features: List[str], output_features: str) -> str:
    """Merges multiple feature classes into a single feature class.

    Args:
        input_features: A list of paths to the input feature classes.
        output_features: The path to the output feature class.
    """
    try:
        for fc in input_features:
            if not _feature_class_exists(fc):
                return f"Error: Input feature class '{fc}' does not exist."
        arcpy.management.Merge(input_features, output_features)
        return f"Successfully merged features into '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error merging features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def append_features(input_features: List[str], target_features: str, schema_type: str = "NO_TEST") -> str:
    """Appends features from one or more feature classes to an existing target feature class.

    Args:
        input_features: A list of paths to the input feature classes.
        target_features: The path to the existing target feature class.
        schema_type: How to handle schema differences ("TEST", "NO_TEST"). Default: "NO_TEST".
    """
    try:
        for fc in input_features:
            if not _feature_class_exists(fc):
                return f"Error: Input feature class '{fc}' does not exist."
        if not _feature_class_exists(target_features):
            return f"Error: Target feature class '{target_features}' does not exist."

        if schema_type.upper() not in ("TEST", "NO_TEST"):
            return "Error: Invalid value for 'schema_type'. Must be 'TEST' or 'NO_TEST'."

        arcpy.management.Append(input_features, target_features, schema_type)
        return f"Successfully appended features to '{target_features}'."
    except arcpy.ExecuteError:
        return f"Error appending features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def delete_features(input_features: str) -> str:
    """Deletes a feature class or table.

    Args:
        input_features: The path to the feature class or table to delete.
    """
    try:
        if not _dataset_exists(input_features):
            return f"Error: Input dataset '{input_features}' does not exist."
        arcpy.management.Delete(input_features)
        return f"Successfully deleted '{input_features}'."
    except arcpy.ExecuteError:
        return f"Error deleting features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def create_feature_class(out_path: str, out_name: str, geometry_type: str, template: str = "", spatial_reference: str = "") -> str:
    """Creates a new feature class.

    Args:
        out_path: The path to the geodatabase or folder where the feature class will be created.
        out_name: The name of the new feature class.
        geometry_type: The geometry type (e.g., "POLYGON", "POINT", "POLYLINE").
        template: An optional template feature class (for schema).
        spatial_reference: An optional spatial reference (e.g., "WGS 1984").
    """
    try:
        if not arcpy.Exists(out_path):
            return f"Error: Output path '{out_path}' does not exist."
        if geometry_type.upper() not in ("POINT", "MULTIPOINT", "POLYGON", "POLYLINE", "MULTIPATCH"):
            return "Error: Invalid geometry_type. Must be 'POINT', 'MULTIPOINT', 'POLYGON', 'POLYLINE', or 'MULTIPATCH'."
        if template and not _feature_class_exists(template):
            return f"Error: Template feature class '{template}' does not exist."

        if spatial_reference:
            try:
                sr = arcpy.SpatialReference(spatial_reference)
            except:
                return f"Error: Invalid spatial reference '{spatial_reference}'."
        else:
            sr = None

        arcpy.management.CreateFeatureclass(out_path, out_name, geometry_type.upper(), template, spatial_reference=sr)
        return f"Successfully created feature class '{out_name}' in '{out_path}'."
    except arcpy.ExecuteError:
        return f"Error creating feature class: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def add_field(in_table: str, field_name: str, field_type: str, field_length: int = None, 
              field_precision: int = None, field_scale: int = None) -> str:
    """Adds a new field to a feature class or table.

    Args:
        in_table: The path to the feature class or table.
        field_name: The name of the new field.
        field_type: The data type of the new field (e.g., "TEXT", "SHORT", "LONG", "FLOAT", "DOUBLE", "DATE").
        field_length: The length of the field (for TEXT fields).
        field_precision: The precision of the field (for numeric fields).
        field_scale: The scale of the field (for numeric fields).
    """
    try:
        if not _dataset_exists(in_table):
            return f"Error: Input table/feature class '{in_table}' does not exist."
        if _is_valid_field_name(in_table, field_name):
            return f"Error: Field '{field_name}' already exists in '{in_table}'."

        valid_types = ["TEXT", "SHORT", "LONG", "FLOAT", "DOUBLE", "DATE", "BLOB", "RASTER", "GUID"]
        if field_type.upper() not in valid_types:
            return f"Error: Invalid field_type. Must be one of: {', '.join(valid_types)}"

        arcpy.management.AddField(in_table, field_name, field_type.upper(), field_precision, field_scale, field_length)
        return f"Successfully added field '{field_name}' to '{in_table}'."
    except arcpy.ExecuteError:
        return f"Error adding field: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def calculate_field(in_table: str, field_name: str, expression: str, expression_type: str = "PYTHON3") -> str:
    """Calculates values for a field in a feature class or table.

    Args:
        in_table: The path to the feature class or table.
        field_name: The name of the field to calculate.
        expression: The expression to use for calculation.
        expression_type: The type of expression ("PYTHON3", "ARCADE", "SQL"). Default: "PYTHON3".
    """
    try:
        if not _dataset_exists(in_table):
            return f"Error: Input table/feature class '{in_table}' does not exist."
        if not _is_valid_field_name(in_table, field_name):
            return f"Error: Field '{field_name}' does not exist in '{in_table}'."

        valid_types = ["PYTHON3", "ARCADE", "SQL"]
        if expression_type.upper() not in valid_types:
            return f"Error: Invalid expression_type. Must be one of: {', '.join(valid_types)}"

        arcpy.management.CalculateField(in_table, field_name, expression, expression_type)
        return f"Successfully calculated field '{field_name}' in '{in_table}'."
    except arcpy.ExecuteError:
        return f"Error calculating field: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def select_features(input_features: str, output_features: str, where_clause: str) -> str:
    """Selects features by attribute and saves them to a new feature class. Use the field name not the alias name to make the query.

    Args:
        input_features: The path to the input feature class.
        output_features: The path to the output feature class.
        where_clause: The SQL WHERE clause to select features.
    """
    try:
        if not _feature_class_exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."
        arcpy.analysis.Select(input_features, output_features, where_clause)
        return f"Successfully selected features from '{input_features}' to '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error selecting features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def project_features(input_features: str, output_features: str, out_coor_system: str) -> str:
    """Projects a feature class to a new coordinate system.

    Args:
        input_features: The path to the input feature class.
        output_features: The path to the output feature class.
        out_coor_system: The output coordinate system (e.g., a WKID, name, or path to a .prj file).
    """
    try:
        if not _feature_class_exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."
        try:
            sr = arcpy.SpatialReference(out_coor_system)
        except:
            return f"Error: Invalid output coordinate system '{out_coor_system}'."

        arcpy.management.Project(input_features, output_features, sr)
        return f"Successfully projected '{input_features}' to '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error projecting features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def repair_geometry(input_features: str) -> str:
    """Repairs the geometry of a feature class.

    Args:
        input_features: The path to the input feature class.
    """
    try:
        if not _feature_class_exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."
        arcpy.management.RepairGeometry(input_features)
        return f"Successfully repaired geometry for '{input_features}'."
    except arcpy.ExecuteError:
        return f"Error repairing geometry: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def intersect_features(input_features: List[str], output_features: str, join_attributes: str = "ALL") -> str:
    """Computes the geometric intersection of multiple feature classes.

    Args:
        input_features: A list of paths to the input feature classes.
        output_features: The path to the output feature class.
        join_attributes: Which attributes to join ("ALL", "NO_FID", "ONLY_FID"). Default "ALL".
    """
    try:
        for fc in input_features:
            if not _feature_class_exists(fc):
                return f"Error: Input feature class '{fc}' does not exist."
        if join_attributes.upper() not in ("ALL", "NO_FID", "ONLY_FID"):
            return "Error: Invalid value for join_attributes. Must be 'ALL', 'NO_FID' or 'ONLY_FID'."
        arcpy.analysis.Intersect(input_features, output_features, join_attributes)
        return f"Successfully intersected features to '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error intersecting features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def union_features(input_features: List[str], output_features: str) -> str:
    """Computes the geometric union of multiple feature classes.

    Args:
        input_features: A list of paths to the input feature classes.
        output_features: The path to the output feature class.
    """
    try:
        for fc in input_features:
            if not _feature_class_exists(fc):
                return f"Error: Input feature class '{fc}' does not exist."
        arcpy.analysis.Union(input_features, output_features)
        return f"Successfully unioned features to '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error unioning features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def erase_features(input_features: str, erase_features: str, output_features: str) -> str:
    """Erases parts of the input features that overlap with the erase features.

    Args:
        input_features: The path to the input feature class.
        erase_features: The path to the feature class used to erase from the input.
        output_features: The path to the output feature class.
    """
    try:
        if not _feature_class_exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."
        if not _feature_class_exists(erase_features):
            return f"Error: Erase feature class '{erase_features}' does not exist."
        arcpy.analysis.Erase(input_features, erase_features, output_features)
        return f"Successfully erased features from '{input_features}' to '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error erasing features: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def spatial_join(target_features: str, join_features: str, output_features: str, 
                join_operation: str = "JOIN_ONE_TO_ONE", join_type: str = "KEEP_ALL", 
                match_option: str = "INTERSECT") -> str:
    """Performs a spatial join between two feature classes.

    Args:
        target_features: The path to the target feature class.
        join_features: The path to the join feature class.
        output_features: The path to the output feature class.
        join_operation: The join operation ("JOIN_ONE_TO_ONE" or "JOIN_ONE_TO_MANY").
        join_type: The join type ("KEEP_ALL" or "KEEP_COMMON").
        match_option: The spatial match option (e.g., "INTERSECT", "WITHIN_A_DISTANCE", "CONTAINS").
    """
    try:
        if not _feature_class_exists(target_features):
            return f"Error: Target feature class '{target_features}' does not exist."
        if not _feature_class_exists(join_features):
            return f"Error: Join feature class '{join_features}' does not exist."

        if join_operation.upper() not in ("JOIN_ONE_TO_ONE", "JOIN_ONE_TO_MANY"):
            return "Error: Invalid 'join_operation'. Must be 'JOIN_ONE_TO_ONE' or 'JOIN_ONE_TO_MANY'."
        if join_type.upper() not in ("KEEP_ALL", "KEEP_COMMON"):
            return "Error: Invalid 'join_type'. Must be 'KEEP_ALL' or 'KEEP_COMMON'."

        arcpy.analysis.SpatialJoin(target_features, join_features, output_features, 
                                 join_operation, join_type, match_option=match_option)
        return f"Successfully performed spatial join to '{output_features}'."
    except arcpy.ExecuteError:
        return f"Error performing spatial join: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def zonal_statistics(in_zone_data: str, zone_field: str, in_value_raster: str, out_table: str,
                    statistics_type: str = "MEAN") -> str:
    """Calculates zonal statistics for each zone in a zone dataset.

    Args:
        in_zone_data: The path to the input zone dataset (feature class or raster).
        zone_field: The field defining the zones.
        in_value_raster: The path to the input value raster.
        out_table: The path to the output table.
        statistics_type: The statistic to calculate (e.g., "MEAN", "SUM", "MINIMUM", "MAXIMUM").
    """
    try:
        if not _dataset_exists(in_zone_data):
            return f"Error: Input zone dataset '{in_zone_data}' does not exist."
        if not _dataset_exists(in_value_raster):
            return f"Error: Input value raster '{in_value_raster}' does not exist."
        if not _is_valid_field_name(in_zone_data, zone_field):
            return f"Error: Zone field is not a valid field"

        valid_stats = ["MEAN", "SUM", "MINIMUM", "MAXIMUM", "RANGE", "STD", "VARIETY", "MAJORITY", "MINORITY", "MEDIAN"]
        if statistics_type.upper() not in valid_stats:
            return f"Error: Invalid statistics_type. Must be one of: {', '.join(valid_stats)}"

        arcpy.sa.ZonalStatisticsAsTable(in_zone_data, zone_field, in_value_raster, out_table, statistics_type=statistics_type.upper())
        return f"Successfully calculated zonal statistics to '{out_table}'."
    except arcpy.ExecuteError:
        return f"Error calculating zonal statistics: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def extract_by_mask(in_raster: str, in_mask_data: str, out_raster: str) -> str:
    """Extracts cells from a raster based on a mask.

    Args:
        in_raster: The path to the input raster.
        in_mask_data: The path to the mask dataset (feature class or raster).
        out_raster: The path to the output raster.
    """
    try:
        if not _dataset_exists(in_raster):
            return f"Error: Input raster '{in_raster}' does not exist."
        if not _dataset_exists(in_mask_data):
            return f"Error: Input mask dataset '{in_mask_data}' does not exist."

        out_raster_obj = arcpy.sa.ExtractByMask(in_raster, in_mask_data)
        out_raster_obj.save(out_raster)
        return f"Successfully extracted raster by mask to '{out_raster}'."
    except arcpy.ExecuteError:
        return f"Error extracting raster by mask: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def slope(in_raster: str, output_raster: str, output_measurement: str = "DEGREE", z_factor: float = 1.0) -> str:
    """Calculates the slope of a raster surface.

    Args:
        in_raster: The path to the input raster.
        output_raster: The path to the output slope raster.
        output_measurement: The units of the output slope ("DEGREE", "PERCENT_RISE").
        z_factor: The Z factor (for vertical exaggeration).
    """
    try:
        if not _dataset_exists(in_raster):
            return f"Error: Input raster '{in_raster}' does not exist."
        if output_measurement.upper() not in ("DEGREE", "PERCENT_RISE"):
            return "Error: Invalid output_measurement. Must be 'DEGREE' or 'PERCENT_RISE'."
        if not isinstance(z_factor, (int, float)):
            return "Error: z_factor must be a number"

        out_slope = arcpy.sa.Slope(in_raster, output_measurement.upper(), z_factor)
        out_slope.save(output_raster)
        return f"Successfully calculated slope to '{output_raster}'."
    except arcpy.ExecuteError:
        return f"Error calculating slope: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def aspect(in_raster: str, output_raster: str) -> str:
    """Calculates the aspect (direction) of a raster surface.

    Args:
        in_raster: The path to the input raster.
        output_raster: The path to the output aspect raster.
    """
    try:
        if not _dataset_exists(in_raster):
            return f"Error: Input raster '{in_raster}' does not exist."

        out_aspect = arcpy.sa.Aspect(in_raster)
        out_aspect.save(output_raster)
        return f"Successfully calculated aspect to '{output_raster}'."
    except arcpy.ExecuteError:
        return f"Error calculating aspect: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def hillshade(in_raster: str, output_raster: str, azimuth: float = 315.0, 
              altitude: float = 45.0, z_factor: float = 1.0) -> str:
    """Calculates the hillshade of a raster surface.

    Args:
        in_raster: The path to the input raster.
        output_raster: The path to the output hillshade raster.
        azimuth: The azimuth angle (0-360).
        altitude: The altitude angle (0-90).
        z_factor: The Z factor (for vertical exaggeration).
    """
    try:
        if not _dataset_exists(in_raster):
            return f"Error: Input raster '{in_raster}' does not exist."

        if not all(isinstance(arg, (int, float)) for arg in [azimuth, altitude, z_factor]):
            return "Error: Azimuth, altitude, and z_factor must be numbers."

        out_hillshade = arcpy.sa.Hillshade(in_raster, azimuth, altitude, "SHADOWS", z_factor)
        out_hillshade.save(output_raster)
        return f"Successfully calculated hillshade to '{output_raster}'."
    except arcpy.ExecuteError:
        return f"Error calculating hillshade: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def reclassify_raster(in_raster: str, reclass_field: str, remap_string: str, output_raster: str,
                      missing_values: str = "DATA") -> str:
    """Reclassifies (substitutes) values in a raster based on a reclassification table.

    Args:
        in_raster: The path to the input raster.
        reclass_field: The field to use for reclassification.
        remap_string: The reclassification mapping string (e.g., "1 5 1;5 10 2;10 20 3").
        output_raster: The path to the output reclassified raster.
        missing_values: How to handle missing values ("DATA" or "NODATA").
    """
    try:
        if not _dataset_exists(in_raster):
            return f"Error: Input raster '{in_raster}' does not exist."

        if missing_values.upper() not in ("DATA", "NODATA"):
            return "Error: Invalid 'missing_values'. Must be 'DATA' or 'NODATA'."

        remap_list = [item.strip().split() for item in remap_string.split(";")]
        remap = arcpy.sa.RemapValue(remap_list)

        out_reclass = arcpy.sa.Reclassify(in_raster, reclass_field, remap, missing_values.upper())
        out_reclass.save(output_raster)
        return f"Successfully reclassified raster to '{output_raster}'."
    except arcpy.ExecuteError:
        return f"Error reclassifying raster: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def define_projection(input_dataset: str, coor_system: str) -> str:
    """Defines the coordinate system for a dataset.

    Args:
        input_dataset: The path to the input dataset.
        coor_system: The coordinate system to define (e.g., a WKID, name, or path to a .prj file).
    """
    try:
        if not _dataset_exists(input_dataset):
            return f"Error: Input dataset '{input_dataset}' does not exist."

        try:
            sr = arcpy.SpatialReference(coor_system)
        except:
            return f"Error: Invalid coordinate system '{coor_system}'."

        arcpy.management.DefineProjection(input_dataset, sr)
        return f"Successfully defined projection for '{input_dataset}'."
    except arcpy.ExecuteError:
        return f"Error defining projection: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def describe_dataset(input_dataset: str) -> str:
    """Describes the properties of a dataset.

    Args:
        input_dataset: The path to the dataset.
    """
    try:
        if not _dataset_exists(input_dataset):
            return f"Error: Input dataset '{input_dataset}' does not exist."

        desc = arcpy.Describe(input_dataset)
        desc_dict = {
            "name": desc.name,
            "dataType": desc.dataType,
        }
        if hasattr(desc, "shapeType"):
            desc_dict["shapeType"] = desc.shapeType
        if hasattr(desc, "spatialReference"):
            desc_dict["spatialReference"] = desc.spatialReference.name

        return json.dumps(desc_dict, indent=4)
    except arcpy.ExecuteError:
        return f"Error describing dataset: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def dataset_exists(dataset: str) -> str:
    """Checks if a dataset exists.

    Args:
        dataset: The path to the dataset.
    """
    try:
        if arcpy.Exists(dataset):
            return "True"
        else:
            return "False"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


@tool
def list_fields(dataset: str) -> str:
    """Lists fields in a feature class or table, including alias names. 

    Args:
        dataset: The path to the feature class or table.
    """
    try:
        if not arcpy.Exists(dataset):
            return f"Error: Input dataset '{dataset}' does not exist."

        fields = arcpy.ListFields(dataset)
        field_info = [{"field_name": f.name, "aliasName": f.aliasName, "type": f.type} for f in fields]
        return json.dumps(field_info, indent=4)
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

    
@tool
def scan_workspace_directory_for_gis_files(workspace: str = None) -> str:
    """Lists all the files: datasets, feature classes, and tables in the current or specified workspace with detailed information.
    
    This tool provides a comprehensive inventory of the ArcGIS workspace contents, including:
    - Datasets with their types and counts
    - Feature classes with their geometry types, spatial references, and feature counts
    - Tables with their row counts
    
    Use it to find out all the gis files in the workspace and their information.
    Args:
        workspace (str, required): Path to the workspace (geodatabase or folder). 
    
    Returns:
        str: A JSON-formatted string containing:
            {
                "datasets": [
                    {
                        "name": str,  # Name of the dataset
                        "type": str,  # Dataset type (e.g., "FeatureDataset", "RasterDataset")
                    }
                ],
                "feature_classes": [
                    {
                        "name": str,  # Name of the feature class
                        "type": str,  # Shape type (e.g., "Polygon", "Point", "Polyline")
                        "spatial_reference": str,  # Spatial reference name
                        "feature_count": int  # Number of features in the feature class
                    }
                ],
                "tables": [
                    {
                        "name": str,  # Name of the table
                        "row_count": int  # Number of rows in the table
                    }
                ]  # List of tables with their row counts
            }
    
    Raises:
        arcpy.ExecuteError: If there's an error accessing the workspace or listing contents
    """

    try:
        if workspace:
            arcpy.env.workspace = workspace
            
        datasets = arcpy.ListDatasets()
        fcs = arcpy.ListFeatureClasses()
        tables = arcpy.ListTables()
        
        inventory = {
            "datasets": [
                {
                    "name": ds,
                    "type": arcpy.Describe(ds).datasetType
                } for ds in (datasets or [])
            ],
            "feature_classes": [
                {
                    "name": fc,
                    "type": arcpy.Describe(fc).shapeType,
                    "spatial_reference": arcpy.Describe(fc).spatialReference.name,
                    "feature_count": arcpy.GetCount_management(fc)[0]  # Count of features
                } for fc in (fcs or [])
            ],
            "tables": [
                {
                    "name": table,
                    "row_count": arcpy.GetCount_management(table)[0]  # Count of rows in the table
                } for table in (tables or [])
            ]
        }
        return json.dumps(inventory, indent=2)
    except arcpy.ExecuteError:
        return f"ArcGIS Error: {arcpy.GetMessages(2)}"

@tool
def get_environment_settings(environment_setting: str) -> str:
    """Retrieves the value of a specified ArcGIS environment setting.

    Args:
        environment_setting: The name of the environment setting to retrieve.
    """
    try:
        setting_value = getattr(arcpy.env, environment_setting)
        return str(setting_value)
    except AttributeError:
        return f"Error: Environment setting '{environment_setting}' not found."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def create_file_geodatabase(out_folder_path: str, out_name: str) -> str:
    """Creates a new file geodatabase.

    Args:
        out_folder_path: The path to the folder where the geodatabase will be created.
        out_name: The name of the new file geodatabase (without the .gdb extension).
    """
    try:
        if not os.path.exists(out_folder_path):
            return f"Error: Output folder '{out_folder_path}' does not exist."
        arcpy.management.CreateFileGDB(out_folder_path, out_name)
        return f"Successfully created file geodatabase '{out_name}.gdb' in '{out_folder_path}'."
    except arcpy.ExecuteError:
        return f"Error creating file geodatabase: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def create_feature_dataset(out_dataset_path: str, out_name: str, spatial_reference: str) -> str:
    """Creates a new feature dataset within a geodatabase.

    Args:
        out_dataset_path: The path to the geodatabase where the feature dataset will be created.
        out_name: The name of the new feature dataset.
        spatial_reference: The spatial reference of the feature dataset.
    """
    try:
        if not arcpy.Exists(out_dataset_path):
            return f"Error: Output geodatabase '{out_dataset_path}' does not exist."

        try:
            sr = arcpy.SpatialReference(spatial_reference)
        except:
            return f"Error: Invalid spatial reference '{spatial_reference}'."

        arcpy.management.CreateFeatureDataset(out_dataset_path, out_name, sr)
        return f"Successfully created feature dataset '{out_name}' in '{out_dataset_path}'."
    except arcpy.ExecuteError:
        return f"Error creating feature dataset: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def import_csv(input_csv: str, output_table: str, x_field: str = None, y_field: str = None, 
               z_field: str = None, spatial_reference: str = None) -> str:
    """Imports a CSV file to an ArcGIS table. Optionally creates points.

    Args:
        input_csv: Path to the input CSV file.
        output_table: Path to the output table.
        x_field: Name of the X coordinate field (if creating points).
        y_field: Name of the Y coordinate field (if creating points).
        z_field: Optional. Name of Z coordinate field.
        spatial_reference: Optional. Spatial reference for point creation.
    """
    try:
        if not os.path.exists(input_csv):
            return f"Error: CSV file not found: {input_csv}"

        arcpy.management.TableToTable(input_csv, os.path.dirname(output_table), os.path.basename(output_table))

        if x_field and y_field:
            if spatial_reference:
                sr = arcpy.SpatialReference(spatial_reference)
            else:
                sr = None

            temp_layer = "temp_layer"
            arcpy.management.XYTableToPoint(output_table, temp_layer, x_field, y_field, z_field, sr)
            arcpy.management.CopyFeatures(temp_layer, output_table.replace(".dbf", "").replace(".csv", "") + "_points")
            arcpy.management.Delete(temp_layer)
            return f"Successfully imported CSV to table '{output_table}' and created point feature class."
        return f"Successfully imported CSV to table '{output_table}'."
    except arcpy.ExecuteError:
        return f"Error importing CSV: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def export_to_csv(input_table: str, output_csv: str) -> str:
    """Exports a feature class or table to a CSV file.

    Args:
        input_table: The path to the input feature class or table.
        output_csv: The path to the output CSV file.
    """
    try:
        if not arcpy.Exists(input_table):
            return f"Error: Input table or feature class '{input_table}' does not exist."

        arcpy.conversion.TableToTable(input_table, os.path.dirname(output_csv), os.path.basename(output_csv))
        return f"Successfully exported '{input_table}' to '{output_csv}'."
    except arcpy.ExecuteError:
        return f"Error exporting to CSV: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def service_area(network_dataset: str, input_facilities: str, output_layer_name: str,
                impedance_attribute: str = "TravelTime", travel_mode: str = "Driving Time",
                cutoffs: str = "5 10 15", travel_direction: str = "FROM_FACILITIES") -> str:
    """Calculates service areas around facilities on a network dataset.

    Args:
        network_dataset: Path to the network dataset.
        input_facilities: Path to the feature class containing facility locations.
        output_layer_name: Name of the output service area layer.
        impedance_attribute: Name of the impedance attribute. Default: "TravelTime".
        travel_mode: Name of the travel mode. Default: "Driving Time".
        cutoffs: Space-separated string of cutoff values.
        travel_direction: "FROM_FACILITIES" or "TO_FACILITIES".
    """
    try:
        if not _dataset_exists(network_dataset):
            return f"Error: Network dataset '{network_dataset}' does not exist."
        if not _feature_class_exists(input_facilities):
            return f"Error: Input facilities feature class '{input_facilities}' does not exist."
        if travel_direction.upper() not in ("FROM_FACILITIES", "TO_FACILITIES"):
            return "Error: Invalid 'travel_direction'. Must be 'FROM_FACILITIES' or 'TO_FACILITIES'."

        solver = arcpy.nax.ServiceArea(network_dataset)
        solver.travelMode = travel_mode
        solver.timeUnits = arcpy.nax.TimeUnits.Minutes
        solver.defaultImpedanceCutoffs = [float(c) for c in cutoffs.split()]
        solver.travelDirection = travel_direction.upper()
        solver.impedance = impedance_attribute

        solver.load(arcpy.nax.ServiceAreaInputDataType.Facilities, input_facilities)
        result = solver.solve()

        if result.solveSucceeded:
            result.export(arcpy.nax.ServiceAreaOutputDataType.Polygons, output_layer_name)
            return f"Successfully created service area layer: '{output_layer_name}'"
        else:
            return f"Error: Service area solve failed. Messages:\n{result.solverMessages(arcpy.nax.MessageSeverity.All)}"
    except arcpy.ExecuteError:
        return f"Error in Service Area analysis: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def closest_facility(network_dataset: str, input_incidents: str, input_facilities: str,
                    output_layer_name: str, impedance_attribute: str = "TravelTime",
                    travel_mode: str = "Driving Time", cutoff: str = None,
                    travel_direction: str = "TO_FACILITIES") -> str:
    """Finds the closest facility to each incident on a network dataset.

    Args:
        network_dataset: Path to the network dataset.
        input_incidents: Path to the feature class containing incident locations.
        input_facilities: Path to the feature class containing facility locations.
        output_layer_name: Name of the output closest facility layer.
        impedance_attribute: Name of the impedance attribute.
        travel_mode: Name of the travel mode.
        cutoff: Optional impedance cutoff value.
        travel_direction: "FROM_FACILITIES" or "TO_FACILITIES".
    """
    try:
        if not _dataset_exists(network_dataset):
            return f"Error: Network dataset '{network_dataset}' does not exist."
        if not _feature_class_exists(input_incidents):
            return f"Error: Input incidents feature class '{input_incidents}' does not exist."
        if not _feature_class_exists(input_facilities):
            return f"Error: Input facilities feature class '{input_facilities}' does not exist."

        if travel_direction.upper() not in ("FROM_FACILITIES", "TO_FACILITIES"):
            return "Error: Invalid 'travel_direction'. Must be 'FROM_FACILITIES' or 'TO_FACILITIES'."

        solver = arcpy.nax.ClosestFacility(network_dataset)
        solver.travelMode = travel_mode
        solver.timeUnits = arcpy.nax.TimeUnits.Minutes
        solver.travelDirection = travel_direction.upper()
        solver.impedance = impedance_attribute
        if cutoff:
            try:
                solver.defaultImpedanceCutoff = float(cutoff)
            except ValueError:
                return "Error: Invalid cutoff value. Must be a number."

        solver.load(arcpy.nax.ClosestFacilityInputDataType.Facilities, input_facilities)
        solver.load(arcpy.nax.ClosestFacilityInputDataType.Incidents, input_incidents)

        result = solver.solve()

        if result.solveSucceeded:
            result.export(arcpy.nax.ClosestFacilityOutputDataType.Routes, output_layer_name)
            return f"Successfully created closest facility layer: '{output_layer_name}'"
        else:
            return f"Error: Closest facility solve failed. Messages:\n{result.solverMessages(arcpy.nax.MessageSeverity.All)}"
    except arcpy.ExecuteError:
        return f"Error in Closest Facility analysis: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def route(network_dataset: str, input_stops: str, output_layer_name: str,
          impedance_attribute: str = "TravelTime", travel_mode: str = "Driving Time",
          find_best_order: bool = False) -> str:
    """Calculates the best route between a set of stops on a network dataset.

    Args:
        network_dataset: Path to the network dataset.
        input_stops: Path to the feature class containing stop locations.
        output_layer_name: Name of the output route layer.
        impedance_attribute: Name of the impedance attribute.
        travel_mode: Name of the travel mode.
        find_best_order: Whether to find the best order of stops.
    """
    try:
        if not _dataset_exists(network_dataset):
            return f"Error: Network dataset '{network_dataset}' does not exist."
        if not _feature_class_exists(input_stops):
            return f"Error: Input stops feature class '{input_stops}' does not exist."

        solver = arcpy.nax.Route(network_dataset)
        solver.travelMode = travel_mode
        solver.timeUnits = arcpy.nax.TimeUnits.Minutes
        solver.impedance = impedance_attribute
        solver.findBestSequence = find_best_order

        solver.load(arcpy.nax.RouteInputDataType.Stops, input_stops)
        result = solver.solve()

        if result.solveSucceeded:
            result.export(arcpy.nax.RouteOutputDataType.Routes, output_layer_name)
            return f"Successfully created route layer: '{output_layer_name}'"
        else:
            return f"Error: Route solve failed. Messages:\n{result.solverMessages(arcpy.nax.MessageSeverity.All)}"
    except arcpy.ExecuteError:
        return f"Error in Route analysis: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def download_landsat_tool(
    output_directory: str,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
    landsat_sensors: List[str] = None,
    bands: List[str] = None,
    aoi_feature_class: str = None,
    bounding_box: str = None,
    delete_archive: bool = True,
    max_concurrent_downloads: int = 5,
) -> str:
    """Downloads Landsat Collection 2 Level-2 imagery (Surface Reflectance/Temperature) asynchronously via the USGS M2M API.

    This function automates the process of searching for, requesting, and downloading Landsat data.
    It handles authentication, scene searching, download option retrieval, batch download requests,
    polling for download URLs, downloading files, and (optionally) extracting tar archives.  It uses
    `asyncio` and `aiohttp` for concurrent downloads, significantly improving download speed.  `nest_asyncio`
    is used to allow for nested event loops, making it suitable for use in Jupyter notebooks and IPython.

    Args:
        output_directory: The directory where downloaded files and extracted data will be stored.
            This directory will be created if it doesn't exist.
        start_date: The start date for the imagery search (inclusive), in 'YYYY-MM-DD' format.
        end_date: The end date for the imagery search (inclusive), in 'YYYY-MM-DD' format.
        max_cloud_cover: The maximum acceptable cloud cover percentage (0-100).  Defaults to 20.0.
        landsat_sensors: A list of Landsat sensor identifiers to search for.  Valid options are
            "L8", "L9", "L7", and "L5". Defaults to ["L8", "L9"] if not specified.
        bands: A list of band identifiers to download.  If None, the entire product bundle is
            downloaded.  Band identifiers should be in the format "B1", "B2", etc. (e.g., ["B2", "B3", "B4"]).
            If specified, individual band files are downloaded instead of the full bundle.
        aoi_feature_class:  Not currently supported.  Reserved for future use with feature classes.
        bounding_box: A string defining the bounding box for the search, in the format
            "min_lon,min_lat,max_lon,max_lat" (WGS84 coordinates).
        delete_archive: If True (default), downloaded tar archives are deleted after extraction
            (only applicable when downloading full bundles, i.e., when `bands` is None).
        max_concurrent_downloads: The maximum number of concurrent downloads. Defaults to 5.  Higher
            values can improve download speed but may overwhelm your system or the server.

    Returns:
        A string summarizing the result of the download process, indicating the number of
        files/scenes downloaded and the output directory.  Returns an error message if any
        part of the process fails."""

    base_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"

    def m2m_request(endpoint: str, payload: dict, apiKey: str = None) -> dict:
        url = base_url + endpoint
        headers = {}
        if apiKey:
            headers["X-Auth-Token"] = apiKey
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        resp_json = response.json()
        if resp_json.get("errorCode"):
            raise Exception(f"{resp_json.get('errorCode', 'Unknown Error')}: {resp_json.get('errorMessage', '')}")
        return resp_json.get("data")

    # --- Input validation and setup ---
    if not all([output_directory, start_date, end_date, bounding_box]):
        return "Error: output_directory, start_date, end_date, and bounding_box are required."
    os.makedirs(output_directory, exist_ok=True)

    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bounding_box.split(","))
        if not (-180 <= min_lon <= 180 and -90 <= min_lat <= 90 and -180 <= max_lon <= 180 and -90 <= max_lat <= 90):
            raise ValueError()
    except (ValueError, TypeError):
        return "Error: Invalid bounding_box format. Use 'min_lon,min_lat,max_lon,max_lat'."

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Error: Invalid date format. Use 'YYYY-MM-DD'."

    if landsat_sensors is None:
        landsat_sensors = ["L8", "L9"]

    # 2. Authentication
    username = os.getenv("EARTHDATA_USER")
    token = os.getenv("EARTHDATA_TOKEN")
    if not username or not token:
        return "Error: EARTHDATA_USER and EARTHDATA_TOKEN environment variables must be set."

    login_payload = {"username": username, "token": token}
    try:
        apiKey = m2m_request("login-token", login_payload)
        print("Login successful.")
    except Exception as e:
        return f"Login failed: {str(e)}"

    # 3. Scene Search
    datasets_map = {
        "L8": "landsat_ot_c2_l2",
        "L9": "landsat_ot_c2_l2",
        "L7": "landsat_etm_c2_l2",
        "L5": "landsat_tm_c2_l2"
    }
    scene_list = []
    for sensor in landsat_sensors:
        sensor_key = sensor.upper()
        if sensor_key not in datasets_map:
            m2m_request("logout", {}, apiKey)
            return f"Error: Invalid sensor '{sensor}'."
        dataset = datasets_map[sensor_key]
        search_payload = {
            "datasetName": dataset,
            "sceneFilter": {
                "spatialFilter": {"filterType": "geojson", "geoJson": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, min_lat], [min_lon, min_lat]
                    ]]
                }},
                "acquisitionFilter": {"start": start_date, "end": end_date},
                "cloudCoverFilter": {"min": 0, "max": int(max_cloud_cover)}
            }
        }
        try:
            search_data = m2m_request("scene-search", search_payload, apiKey)
            results = search_data.get("results", [])
            print(f"Found {len(results)} scenes for sensor {sensor_key}.")
            scene_list.extend([(scene.get("entityId"), dataset, scene.get("displayId"))
                               for scene in results if scene.get("entityId") and scene.get("displayId")])
        except Exception as e:
            m2m_request("logout", {}, apiKey)
            return f"Search failed for sensor {sensor_key}: {str(e)}"

    if not scene_list:
        m2m_request("logout", {}, apiKey)
        return "No scenes found."

    entity_to_display = {entityId: displayId for entityId, _, displayId in scene_list}

    # 4. Download Options
    downloads = []
    dataset_groups: Dict[str, List[str]] = {}
    for entityId, ds, _ in scene_list:
        dataset_groups.setdefault(ds, []).append(entityId)


    if bands is None:
        # Full bundle workflow
        for ds, entityIds in dataset_groups.items():
            payload = {"datasetName": ds, "entityIds": entityIds}
            try:
                dload_data = m2m_request("download-options", payload, apiKey)
                if isinstance(dload_data, list):
                    options = dload_data
                elif isinstance(dload_data, dict):
                    options = dload_data.get("options", [])
                else:
                    options = []
                options_by_entity: Dict[str, dict] = {}
                for opt in options:
                    if opt.get("available"):
                        ent_id = opt.get("entityId")
                        if ent_id and ent_id not in options_by_entity:
                            options_by_entity[ent_id] = opt
                for ent in entityIds:
                    if ent in options_by_entity:
                        opt = options_by_entity[ent]
                        downloads.append({"entityId": ent, "productId": opt.get("id")})
                # Removed: No longer needed for verbose output: print(f"Retrieved bundle download options for dataset {ds}.")
            except Exception as e:
                print(f"Download-options request failed for dataset {ds}: {str(e)}")  # Keep: Important for error handling

    else:  # Specific bands
        for dataset_name, entity_ids in dataset_groups.items():
            payload = {"datasetName": dataset_name, "entityIds": entity_ids}
            try:
                options = m2m_request("download-options", payload, apiKey)
                if isinstance(options, dict):
                    options = options.get("options", [])
                options_by_entity = {opt["entityId"]: opt for opt in options if opt.get("available") and opt.get("entityId")}

                for entity_id in entity_ids:
                    if entity_id in options_by_entity:
                        option = options_by_entity[entity_id]
                        for secondary_option in option.get("secondaryDownloads", []):
                            if secondary_option.get("available"):
                                file_id = secondary_option.get("displayId", "")
                                if any(file_id.endswith(f"_{band_code.upper()}.TIF") for band_code in bands):
                                    downloads.append({"entityId": secondary_option["entityId"], "productId": secondary_option["id"]})
            except Exception as e:
                print(f"Download-options request failed for dataset {dataset_name}: {str(e)}")

    if not downloads:
        m2m_request("logout", {}, apiKey)
        return "No available downloads found."

    # 5. Download Request
    label = datetime.now().strftime("%Y%m%d_%H%M%S")
    req_payload = {"downloads": downloads, "label": label}
    try:
        req_results = m2m_request("download-request", req_payload, apiKey)
        print("Download request submitted.")
        available_downloads = req_results.get("availableDownloads", [])
        preparing_downloads = req_results.get("preparingDownloads", [])
    except Exception as e:
        m2m_request("logout", {}, apiKey)
        return f"Download request failed: {str(e)}"

    # 6. Poll for Download URLs (if needed) - CORRECTED LOGIC
    if preparing_downloads:  # Poll if *anything* is preparing
        retrieve_payload = {"label": label}
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                ret_data = m2m_request("download-retrieve", retrieve_payload, apiKey)
                if ret_data and ret_data.get("available"):
                    available_downloads.extend(ret_data["available"])  # Add new URLs
                    print(f"Attempt {attempt + 1}: Retrieved {len(ret_data['available'])} URLs.")
                    if len(available_downloads) >= len(downloads):
                        break  # Exit loop when we have all URLs
                else:
                    print("No download URLs retrieved yet.")
            except Exception as e:
                print(f"Download-retrieve attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_attempts - 1:
                print(f"Waiting 20 seconds... ({attempt + 1}/{max_attempts})")
                time.sleep(20)


    # 7. Download and Process (ASYNC)
    async def _download_and_process(downloads_to_process):
        async def _download_single_file(session, item, semaphore):
            async with semaphore:
                download_url = item.get("url")
                if not download_url:
                    return None

                try:
                    async with session.get(download_url, timeout=300) as response:
                        response.raise_for_status()

                        if bands is not None:
                            new_fname = item.get("fileName")
                            if not new_fname:
                                entity_id = item.get("entityId")
                                if entity_id:
                                    new_fname = entity_id + ".TIF"
                                else:
                                    print(f"Warning: entityId missing, skipping: {item}")
                                    return None
                        else:
                            entity = item.get("entityId")
                            display_id = entity_to_display.get(entity)
                            new_fname = (display_id + ".tar.gz") if display_id else os.path.basename(urlparse(download_url).path)

                        final_path = os.path.join(output_directory, new_fname)
                        async with aiofiles.open(final_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        print(f"Downloaded: {new_fname}")

                        if bands is None and new_fname.endswith(".tar.gz"):
                            try:
                                with tarfile.open(final_path, "r:*") as tar:
                                    extract_dir = os.path.join(output_directory, entity_to_display.get(item.get("entityId"), ""))
                                    os.makedirs(extract_dir, exist_ok=True)
                                    tar.extractall(path=extract_dir)
                                print(f"Extracted: {new_fname}")
                                if delete_archive:
                                    os.remove(final_path)
                                    print(f"Deleted archive: {final_path}")
                            except Exception as e:
                                print(f"Error extracting {final_path}: {str(e)}")
                        return final_path

                except Exception as e:
                    print(f"Download failed for {download_url}: {str(e)}")
                    return None

        semaphore = asyncio.Semaphore(max_concurrent_downloads)
        async with aiohttp.ClientSession() as session:
            tasks = [_download_single_file(session, item, semaphore) for item in downloads_to_process]
            return await asyncio.gather(*tasks)

    downloaded_files: List[str] = []
    try:
        # ALWAYS process available_downloads
        downloaded_files = asyncio.run(_download_and_process(available_downloads))
        downloaded_files = [path for path in downloaded_files if path is not None]  # Filter out failures

    except RuntimeError as e:
        print(f"RuntimeError: {e}")

    # 8. Logout
    try:
        m2m_request("logout", {}, apiKey)
        print("Logged out.")
    except Exception as e:
        print(f"Logout failed: {str(e)}")

    return f"Successfully downloaded {len(downloaded_files)} files to {output_directory}."


@tool
def scan_external_directory_for_gis_files(directory_path: str) -> str:
    """Scans a directory for GIS-compatible files and returns their details. 
    These directories are external directories that are not part of the workspace (inside the geodatabases).
    
    Args:
        directory_path: Path to the directory to scan
        
    Returns:
        JSON string containing file information categorized by type:
        {
            "vector_files": [
                {
                    "name": str,
                    "path": str,
                    "type": str,  # "Shapefile", "GeoJSON", etc.
                    "driver": str,
                    "layer_count": int,
                    "crs": str
                }
            ],
            "raster_files": [
                {
                    "name": str,
                    "path": str,
                    "type": str,  # "GeoTIFF", "IMG", etc.
                    "dimensions": [width, height],
                    "bands": int,
                    "crs": str
                }
            ]
        }
    """
    try:
        print(f"Scanning directory: {directory_path}")
        if not os.path.exists(directory_path):
            print(f"Directory does not exist: {directory_path}")
            return json.dumps({"error": f"Directory does not exist: {directory_path}"})
        
        # File extensions to look for
        vector_extensions = {'.shp', '.geojson', '.gml', '.kml', '.gpkg', '.json', '.zip'}
        raster_extensions = {'.tif', '.tiff', '.img', '.dem', '.hgt', '.asc', '.jpg', '.jpeg', '.png', '.bmp'}
        
        result = {
            "vector_files": [],
            "raster_files": []
        }
        
        # Count files for logging
        total_files = 0
        vector_files_found = 0
        raster_files_found = 0
        
        print(f"Starting directory walk in {directory_path}")
        for root, dirs, files in os.walk(directory_path):
            print(f"Scanning subdirectory: {root} with {len(files)} files")
            for file in files:
                total_files += 1
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                # Vector files
                if ext in vector_extensions:
                    try:
                        print(f"Attempting to open vector file: {file_path}")
                        with fiona.open(file_path) as src:
                            vector_files_found += 1
                            result["vector_files"].append({
                                "name": file,
                                "path": file_path,
                                "type": src.driver,
                                "driver": src.driver,
                                "layer_count": len(src),
                                "crs": str(src.crs)
                            })
                            print(f"Successfully processed vector file: {file_path}")
                    except Exception as e:
                        print(f"Error processing vector file {file_path}: {str(e)}")
                
                # Raster files
                elif ext in raster_extensions:
                    try:
                        print(f"Attempting to open raster file: {file_path}")
                        with rasterio.open(file_path) as src:
                            raster_files_found += 1
                            result["raster_files"].append({
                                "name": file,
                                "path": file_path,
                                "type": src.driver,
                                "dimensions": [src.width, src.height],
                                "bands": src.count,
                                "crs": str(src.crs)
                            })
                            print(f"Successfully processed raster file: {file_path}")
                    except Exception as e:
                        print(f"Error processing raster file {file_path}: {str(e)}")
        
        print(f"Scan complete for {directory_path}:")
        print(f"  Total files scanned: {total_files}")
        print(f"  Vector files found: {vector_files_found}")
        print(f"  Raster files found: {raster_files_found}")
        
        return json.dumps(result, indent=2)
    except Exception as e:
        error_msg = f"Error scanning directory {directory_path}: {str(e)}"
        print(error_msg)
        traceback.print_exc()  # Print the full traceback for debugging
        return json.dumps({"error": error_msg})

@tool
def raster_calculator(input_rasters: List[str], expression: str, output_raster: str) -> str:
    """Performs calculations on rasters using map algebra expressions.
    
    This tool allows you to perform mathematical operations on raster datasets using map algebra.
    Each input raster is referenced in the expression as 'r1', 'r2', etc., in the order they appear 
    in the input_rasters list. The calculation is performed on a cell-by-cell basis.
    
    GIS Concepts:
    - Map algebra allows mathematical operations on raster cells
    - Operations are performed on a cell-by-cell basis
    - Common uses include terrain analysis, vegetation indices, raster reclassification
    - Can create new derived datasets from existing rasters
    
    Args:
        input_rasters: List of raster paths to use in the calculation. 
                      Each raster will be referenced in the expression as 'r1', 'r2', etc.
                      All rasters must be valid and accessible.
        expression: A valid map algebra expression using the references 'r1', 'r2', etc.
                   Can include mathematical operators (+, -, *, /), functions (Abs, Sin, Cos),
                   and logical operators (<, >, ==, etc.).
        output_raster: Path to save the result raster.
                      Will be overwritten if it already exists.
    
    Returns:
        A message indicating success or failure with details about the operation.
        
    Example:
        >>> raster_calculator(["dem1.tif", "dem2.tif"], "(r1 + r2) / 2", "average_dem.tif")
        "Successfully calculated and saved raster to 'average_dem.tif'."
        
        >>> raster_calculator(["landsat_nir.tif", "landsat_red.tif"], "(r1 - r2) / (r1 + r2)", "ndvi.tif")
        "Successfully calculated and saved raster to 'ndvi.tif'."
    
    Notes:
        - Requires Spatial Analyst extension
        - Expression syntax must be valid Python that evaluates to a raster
        - All input rasters should have the same spatial resolution and extent for best results
        - Output will have the same spatial reference as the inputs
    """
    try:
        # Input validation
        if not input_rasters:
            return "Error: No input rasters provided."
            
        # Validate that all input rasters exist
        for i, raster in enumerate(input_rasters):
            if not _validate_raster_exists(raster):
                return f"Error: Input raster '{raster}' does not exist or is not a valid raster."
        
        # Create references to the input rasters
        rasters = {}
        for i, raster_path in enumerate(input_rasters):
            raster_var = f"r{i+1}"
            rasters[raster_var] = arcpy.Raster(raster_path)
        
        # Replace raster references in the expression
        calc_expression = expression
        for var_name, raster_obj in rasters.items():
            # This approach preserves the expression as is, letting arcpy's Raster Calculator handle it
            locals()[var_name] = raster_obj
        
        # Execute the calculation using the spatial analyst extension
        arcpy.CheckOutExtension("Spatial")
        try:
            # Use eval to create the raster algebra expression
            result_raster = eval(calc_expression)
            result_raster.save(output_raster)
            arcpy.CheckInExtension("Spatial")
            return f"Successfully calculated and saved raster to '{output_raster}'."
        except SyntaxError:
            return f"Error: Invalid expression syntax: {expression}"
        except Exception as e:
            return f"Error in raster calculation: {str(e)}"
        finally:
            arcpy.CheckInExtension("Spatial")
    except arcpy.ExecuteError:
        return f"ArcPy error: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def spatial_autocorrelation(input_features: str, value_field: str,
                           contiguity_method: str = "INVERSE_DISTANCE") -> str:
    """Calculates Moran's I spatial autocorrelation statistic to measure spatial patterns.
    
    This tool analyzes whether a set of features and their associated values exhibit clustered,
    dispersed, or random spatial patterns. It calculates Moran's I, a measure of spatial 
    autocorrelation, along with z-scores and p-values to indicate statistical significance.
    
    GIS Concepts:
    - Spatial autocorrelation measures how objects are related in space
    - Positive spatial autocorrelation: similar values cluster together
    - Negative spatial autocorrelation: dissimilar values cluster together
    - Random pattern: no spatial relationship between values
    - Statistical significance determines if patterns are likely due to random chance
    
    Args:
        input_features: The path to the input feature class containing the values to analyze.
                       Must be an existing feature class.
        value_field: The field containing the numeric values to analyze for spatial patterns.
                    Must be a valid field in the input feature class with numeric values.
        contiguity_method: The conceptualization of spatial relationships between features.
                          Options include:
                          - "INVERSE_DISTANCE" (default): Nearby neighbors have larger influence
                          - "FIXED_DISTANCE_BAND": Features within a critical distance are analyzed
                          - "K_NEAREST_NEIGHBORS": Only the k nearest features are analyzed
                          - "CONTIGUITY_EDGES_ONLY": Only features that share an edge are analyzed
                          - "CONTIGUITY_EDGES_CORNERS": Features sharing edge or corner are analyzed
                          - "GET_SPATIAL_WEIGHTS_FROM_FILE": Uses an external spatial weights file
    
    Returns:
        A JSON string containing:
        - Moran's I index value: ranges from -1 (dispersed) to +1 (clustered)
        - z-score: standard deviations from the mean
        - p-value: probability the observed pattern is random
        - Interpretation of the results
        
    Example:
        >>> spatial_autocorrelation("census_tracts.shp", "income", "INVERSE_DISTANCE")
        {
          "moran_index": 0.75,
          "z_score": 4.2,
          "p_value": 0.00001,
          "result_type": "Moran's I",
          "interpretation": "The pattern exhibits statistically significant clustering."
        }
    
    Notes:
        - Requires a feature class with at least 30 features for reliable results
        - Different contiguity methods may produce different results
        - High positive z-scores indicate clustering of similar values
        - High negative z-scores indicate dispersion of similar values
        - P-values less than 0.05 typically indicate statistical significance
    """
    try:
        # Validate inputs
        if not _feature_class_exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."
        
        if not _is_valid_field_name(input_features, value_field):
            return f"Error: Field '{value_field}' does not exist in '{input_features}'."
        
        valid_methods = ["INVERSE_DISTANCE", "FIXED_DISTANCE_BAND", "K_NEAREST_NEIGHBORS", 
                         "CONTIGUITY_EDGES_ONLY", "CONTIGUITY_EDGES_CORNERS", "GET_SPATIAL_WEIGHTS_FROM_FILE"]
        
        if contiguity_method.upper() not in valid_methods:
            return f"Error: Invalid contiguity_method. Must be one of: {', '.join(valid_methods)}"
        
        # Execute Spatial Autocorrelation (Moran's I)
        result = arcpy.stats.SpatialAutocorrelation(
            input_features, 
            value_field, 
            contiguity_method.upper(),
            "NO_STANDARDIZATION", 
            "EUCLIDEAN_DISTANCE",
            "NONE")
        
        # Extract the results
        results_dict = {
            "moran_index": result.getOutput(0),
            "z_score": result.getOutput(1),
            "p_value": result.getOutput(2),
            "result_type": "Moran's I",
            "interpretation": _interpret_moran_i(float(result.getOutput(1)), float(result.getOutput(2)))
        }
        
        return json.dumps(results_dict, indent=2)
    except arcpy.ExecuteError:
        return f"Error in Spatial Autocorrelation: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def _interpret_moran_i(z_score: float, p_value: float) -> str:
    """Helper function to interpret Moran's I results."""
    if p_value > 0.05:
        return "The pattern does not appear to be significantly different than random."
    elif z_score > 0:
        return "The pattern exhibits statistically significant clustering."
    else:
        return "The pattern exhibits statistically significant dispersion."

@tool
def nearest_neighbor_analysis(input_features: str, distance_method: str = "EUCLIDEAN") -> str:
    """Calculate nearest neighbor statistics for point patterns.
    
    Args:
        input_features: The path to the input point feature class.
        distance_method: The method for calculating distances.
                        Options include "EUCLIDEAN" or "MANHATTAN_DISTANCE".
    
    Returns:
        A JSON string containing the nearest neighbor ratio, z-score, and p-value.
    """
    try:
        # Validate inputs
        if not _feature_class_exists(input_features):
            return f"Error: Input feature class '{input_features}' does not exist."
        
        # Check if the input is a point feature class
        desc = arcpy.Describe(input_features)
        if desc.shapeType != "Point":
            return f"Error: Input feature class must be of point type. Current type: {desc.shapeType}"
        
        valid_methods = ["EUCLIDEAN", "MANHATTAN_DISTANCE"]
        if distance_method.upper() not in valid_methods:
            return f"Error: Invalid distance_method. Must be one of: {', '.join(valid_methods)}"
        
        # Execute Average Nearest Neighbor
        result = arcpy.stats.AverageNearestNeighbor(
            input_features,
            distance_method.upper(),
            "EUCLIDEAN_DISTANCE",
            "NONE")
        
        # Extract the results
        results_dict = {
            "nearest_neighbor_ratio": result.getOutput(0),
            "z_score": result.getOutput(1),
            "p_value": result.getOutput(2),
            "observed_mean_distance": result.getOutput(3),
            "expected_mean_distance": result.getOutput(4),
            "interpretation": _interpret_nearest_neighbor(float(result.getOutput(0)), float(result.getOutput(1)), float(result.getOutput(2)))
        }
        
        return json.dumps(results_dict, indent=2)
    except arcpy.ExecuteError:
        return f"Error in Nearest Neighbor Analysis: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def _interpret_nearest_neighbor(nn_ratio: float, z_score: float, p_value: float) -> str:
    """Helper function to interpret nearest neighbor results."""
    if p_value > 0.05:
        return "The pattern does not appear to be significantly different than random."
    elif nn_ratio < 1:
        return "The pattern exhibits statistically significant clustering."
    else:
        return "The pattern exhibits statistically significant dispersion."

@tool
def calculate_ndvi(nir_band: str, red_band: str, output_raster: str) -> str:
    """Calculates Normalized Difference Vegetation Index (NDVI) from NIR and Red bands.
    
    NDVI is one of the most common vegetation indices used to assess vegetation health and density.
    The formula is: NDVI = (NIR - Red) / (NIR + Red)
    
    NDVI values range from -1 to 1, where:
    - Higher values (0.6 to 1.0): Dense, healthy vegetation
    - Moderate values (0.2 to 0.5): Sparse vegetation
    - Low values (0 to 0.1): Bare soil, rocks, urban areas
    - Negative values: Water, snow, clouds
    
    GIS Concepts:
    - NDVI uses the contrast between red light absorption and NIR reflection by vegetation
    - Healthy plants absorb red light for photosynthesis and reflect NIR
    - NDVI is useful for monitoring vegetation health, drought, agricultural productivity
    - It can help identify seasonal changes and long-term vegetation trends
    
    Args:
        nir_band: The path to the NIR (Near Infrared) band raster.
                 For Landsat 8-9, this is typically Band 5.
                 For Sentinel-2, this is typically Band 8.
        red_band: The path to the Red band raster.
                 For Landsat 8-9, this is typically Band 4.
                 For Sentinel-2, this is typically Band 4.
        output_raster: The path to save the output NDVI raster.
                      Will be overwritten if it already exists.
    
    Returns:
        A message indicating success or failure with details about the operation.
        
    Example:
        >>> calculate_ndvi("landsat8_B5.tif", "landsat8_B4.tif", "ndvi_result.tif")
        "Successfully calculated NDVI and saved to 'ndvi_result.tif'."
    
    Notes:
        - Requires Spatial Analyst extension
        - Input bands should be from the same image/date
        - Both bands should have the same spatial resolution and extent
        - Output values range from -1 to 1, with higher values indicating healthier vegetation
        - Common transformations include scaling to 0-255 or 0-100 for visualization
    """
    try:
        # Validate inputs
        if not _validate_raster_exists(nir_band):
            return f"Error: NIR band raster '{nir_band}' does not exist or is not a valid raster."
        
        if not _validate_raster_exists(red_band):
            return f"Error: Red band raster '{red_band}' does not exist or is not a valid raster."
        
        # Execute NDVI calculation
        arcpy.CheckOutExtension("Spatial")
        try:
            nir = arcpy.Raster(nir_band)
            red = arcpy.Raster(red_band)
            
            # Calculate NDVI: (NIR - Red) / (NIR + Red)
            ndvi = arcpy.sa.Float(nir - red) / arcpy.sa.Float(nir + red)
            ndvi.save(output_raster)
            
            arcpy.CheckInExtension("Spatial")
            return f"Successfully calculated NDVI and saved to '{output_raster}'."
        except Exception as e:
            return f"Error in NDVI calculation: {str(e)}"
        finally:
            arcpy.CheckInExtension("Spatial")
    except arcpy.ExecuteError:
        return f"ArcPy error: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def calculate_savi(nir_band: str, red_band: str, output_raster: str,
                  soil_factor: float = 0.5) -> str:
    """Calculate Soil-Adjusted Vegetation Index (SAVI).
    
    SAVI = [(NIR - Red) / (NIR + Red + L)] * (1 + L)
    where L is a soil adjustment factor (usually 0.5).
    
    SAVI is similar to NDVI but accounts for soil brightness variations.
    
    Args:
        nir_band: The path to the NIR band raster.
        red_band: The path to the Red band raster.
        output_raster: The path to the output SAVI raster.
        soil_factor: The soil adjustment factor (L). Default is 0.5.
                    Values range from 0 (high vegetation) to 1 (low vegetation).
    
    Returns:
        A message indicating success or failure.
    """
    try:
        # Validate inputs
        if not _validate_raster_exists(nir_band):
            return f"Error: NIR band raster '{nir_band}' does not exist or is not a valid raster."
        
        if not _validate_raster_exists(red_band):
            return f"Error: Red band raster '{red_band}' does not exist or is not a valid raster."
        
        if not isinstance(soil_factor, (int, float)) or soil_factor < 0 or soil_factor > 1:
            return "Error: soil_factor must be a number between 0 and 1."
        
        # Execute SAVI calculation
        arcpy.CheckOutExtension("Spatial")
        try:
            nir = arcpy.Raster(nir_band)
            red = arcpy.Raster(red_band)
            
            # Calculate SAVI: [(NIR - Red) / (NIR + Red + L)] * (1 + L)
            numerator = nir - red
            denominator = nir + red + soil_factor
            savi = arcpy.sa.Float(numerator) / arcpy.sa.Float(denominator) * (1 + soil_factor)
            savi.save(output_raster)
            
            arcpy.CheckInExtension("Spatial")
            return f"Successfully calculated SAVI and saved to '{output_raster}'."
        except Exception as e:
            return f"Error in SAVI calculation: {str(e)}"
        finally:
            arcpy.CheckInExtension("Spatial")
    except arcpy.ExecuteError:
        return f"ArcPy error: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def calculate_tpi(dem_raster: str, output_raster: str, neighborhood_size: int = 3) -> str:
    """Calculate Topographic Position Index (TPI).
    
    TPI compares the elevation of each cell in a DEM to the mean elevation of a specified neighborhood around that cell.
    Positive values indicate locations higher than their surroundings (ridges, peaks),
    negative values indicate locations lower than their surroundings (valleys, pits),
    and values near zero indicate flat areas or areas of constant slope.
    
    Args:
        dem_raster: The path to the input Digital Elevation Model (DEM) raster.
        output_raster: The path to the output TPI raster.
        neighborhood_size: The size of the neighborhood in cells. Default is 3 (3x3 neighborhood).
    
    Returns:
        A message indicating success or failure.
    """
    try:
        # Validate inputs
        if not _validate_raster_exists(dem_raster):
            return f"Error: DEM raster '{dem_raster}' does not exist or is not a valid raster."
        
        if not isinstance(neighborhood_size, int) or neighborhood_size < 3 or neighborhood_size % 2 == 0:
            return "Error: neighborhood_size must be an odd integer greater than or equal to 3."
        
        # Execute TPI calculation
        arcpy.CheckOutExtension("Spatial")
        try:
            # Create a neighborhood object
            neighborhood = arcpy.sa.NbrRectangle(neighborhood_size, neighborhood_size, "CELL")
            
            # Calculate focal statistics (mean elevation in the neighborhood)
            mean_elevation = arcpy.sa.FocalStatistics(
                dem_raster,
                neighborhood,
                "MEAN",
                "NODATA")
            
            # Calculate TPI: Original value - neighborhood mean
            dem = arcpy.Raster(dem_raster)
            tpi = dem - mean_elevation
            
            tpi.save(output_raster)
            
            arcpy.CheckInExtension("Spatial")
            return f"Successfully calculated TPI and saved to '{output_raster}'."
        except Exception as e:
            return f"Error in TPI calculation: {str(e)}"
        finally:
            arcpy.CheckInExtension("Spatial")
    except arcpy.ExecuteError:
        return f"ArcPy error: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def batch_process_features(input_folder: str, output_folder: str,
                         tool_name: str, tool_parameters: Dict[str, Any] = None) -> str:
    """Processes multiple feature classes using a specified GIS tool.
    
    This tool automates the execution of another GIS tool on multiple feature classes.
    It finds all feature classes in a folder and applies the specified tool to each one, 
    saving the results to a new folder. This is useful for processing large datasets or 
    performing repetitive operations across multiple files.
    
    GIS Concepts:
    - Batch processing automates repetitive GIS operations
    - Useful for applying the same analysis to multiple datasets
    - Maintains consistent naming conventions for outputs
    - Significantly reduces manual processing time
    - Ensures consistent tool parameters across multiple datasets
    
    Args:
        input_folder: Path to the folder containing input feature classes.
                     All feature classes in this folder will be processed.
        output_folder: Path to save the output feature classes.
                      Will be created if it doesn't exist.
        tool_name: Name of the tool to use for processing.
                  Must be a valid tool name from this module.
                  Examples: "buffer_features", "clip_features", "project_features"
        tool_parameters: Dictionary of parameters to pass to the tool,
                        excluding input and output paths which are handled automatically.
                        The keys should match the parameter names of the target tool.
                        Example: {"buffer_distance": 100, "buffer_unit": "Meters"}
    
    Returns:
        A message summarizing the batch processing results, including success/failure 
        for each feature class and any error messages.
        
    Example:
        >>> batch_process_features(
        ...     "D:/data/cities", 
        ...     "D:/output/buffered_cities",
        ...     "buffer_features", 
        ...     {"buffer_distance": 1000, "buffer_unit": "Meters"}
        ... )
        "Batch processing summary for 3 feature classes:
        Processed 'cities_usa.shp': Successfully buffered...
        Processed 'cities_canada.shp': Successfully buffered...
        Processed 'cities_mexico.shp': Successfully buffered..."
    
    Notes:
        - The tool will process all feature classes in the input folder
        - Each output will be named with "_processed" appended to the original name
        - If the output folder doesn't exist, it will be created
        - Processing continues even if some feature classes fail
        - A summary of all operations is returned when complete
    """
    try:
        # Validate inputs
        if not os.path.exists(input_folder):
            return f"Error: Input folder '{input_folder}' does not exist."
        
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                return f"Error creating output folder: {str(e)}"
        
        # Check if the tool exists in the module namespace
        if tool_name not in globals() or not callable(globals()[tool_name]):
            return f"Error: Tool '{tool_name}' not found or is not callable."
        
        tool_function = globals()[tool_name]
        
        # Initialize parameters if not provided
        if tool_parameters is None:
            tool_parameters = {}
        
        # Set the environment workspace
        arcpy.env.workspace = input_folder
        
        # List all feature classes in the folder
        feature_classes = []
        for ext in ["shp", "gdb"]:
            if ext == "shp":
                feature_classes.extend(arcpy.ListFeatureClasses(f"*.{ext}"))
            elif ext == "gdb":
                workspace_type = arcpy.Describe(input_folder).workspaceType
                if workspace_type in ["LocalDatabase", "RemoteDatabase"]:
                    feature_classes.extend(arcpy.ListFeatureClasses())
        
        if not feature_classes:
            return f"No feature classes found in '{input_folder}'."
        
        results = []
        for fc in feature_classes:
            input_path = os.path.join(input_folder, fc)
            output_name = f"{os.path.splitext(fc)[0]}_processed{os.path.splitext(fc)[1]}"
            output_path = os.path.join(output_folder, output_name)
            
            # Prepare parameters for the tool
            params = tool_parameters.copy()
            params["input_features"] = input_path
            
            # Handle different parameter naming conventions
            if "output_features" in [param for param in tool_function.__code__.co_varnames]:
                params["output_features"] = output_path
            elif "output_feature_class" in [param for param in tool_function.__code__.co_varnames]:
                params["output_feature_class"] = output_path
            else:
                # Add a generic output parameter if the exact name is unknown
                params["output"] = output_path
            
            try:
                # Execute the tool
                result = tool_function(**params)
                results.append(f"Processed '{fc}': {result}")
            except Exception as e:
                results.append(f"Failed to process '{fc}': {str(e)}")
        
        # Format the results
        summary = "\n".join(results)
        return f"Batch processing summary for {len(feature_classes)} feature classes:\n{summary}"
    except arcpy.ExecuteError:
        return f"ArcPy error: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def batch_process_rasters(input_folder: str, output_folder: str,
                         tool_name: str, tool_parameters: Dict[str, Any] = None) -> str:
    """Process multiple rasters using specified tool.
    
    Args:
        input_folder: Path to the folder containing input rasters.
        output_folder: Path to save the output rasters.
        tool_name: Name of the tool to use for processing.
        tool_parameters: Dictionary of parameters to pass to the tool,
                        excluding input and output paths which are handled automatically.
    
    Returns:
        A message summarizing the batch processing results.
    """
    try:
        # Validate inputs
        if not os.path.exists(input_folder):
            return f"Error: Input folder '{input_folder}' does not exist."
        
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                return f"Error creating output folder: {str(e)}"
        
        # Check if the tool exists in the module namespace
        if tool_name not in globals() or not callable(globals()[tool_name]):
            return f"Error: Tool '{tool_name}' not found or is not callable."
        
        tool_function = globals()[tool_name]
        
        # Initialize parameters if not provided
        if tool_parameters is None:
            tool_parameters = {}
        
        # Set the environment workspace
        arcpy.env.workspace = input_folder
        
        # List all rasters in the folder
        rasters = []
        for ext in ["tif", "img", "dem", "asc"]:
            rasters.extend(arcpy.ListRasters(f"*.{ext}"))
        
        if not rasters:
            return f"No rasters found in '{input_folder}'."
        
        results = []
        for raster in rasters:
            input_path = os.path.join(input_folder, raster)
            output_name = f"{os.path.splitext(raster)[0]}_processed{os.path.splitext(raster)[1]}"
            output_path = os.path.join(output_folder, output_name)
            
            # Prepare parameters for the tool
            params = tool_parameters.copy()
            
            # Handle different parameter naming conventions for raster tools
            if "in_raster" in [param for param in tool_function.__code__.co_varnames]:
                params["in_raster"] = input_path
            elif "input_raster" in [param for param in tool_function.__code__.co_varnames]:
                params["input_raster"] = input_path
            else:
                # Add a generic input parameter if the exact name is unknown
                params["input"] = input_path
            
            if "out_raster" in [param for param in tool_function.__code__.co_varnames]:
                params["out_raster"] = output_path
            elif "output_raster" in [param for param in tool_function.__code__.co_varnames]:
                params["output_raster"] = output_path
            else:
                # Add a generic output parameter if the exact name is unknown
                params["output"] = output_path
            
            try:
                # Execute the tool
                result = tool_function(**params)
                results.append(f"Processed '{raster}': {result}")
            except Exception as e:
                results.append(f"Failed to process '{raster}': {str(e)}")
        
        # Format the results
        summary = "\n".join(results)
        return f"Batch processing summary for {len(rasters)} rasters:\n{summary}"
    except arcpy.ExecuteError:
        return f"ArcPy error: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def idw_interpolation(input_points: str, z_field: str, output_raster: str,
                     cell_size: float, power: float = 2) -> str:
    """Performs Inverse Distance Weighted (IDW) interpolation on point data.
    
    IDW interpolation estimates cell values in a raster by averaging the values of sample data 
    points within a specified neighborhood. The influence of each point diminishes with distance 
    according to the power parameter.
    
    GIS Concepts:
    - IDW assumes that points closer to the prediction location have more influence
    - The power parameter controls how quickly influence diminishes with distance
    - Higher power values (>2) increase the influence of the nearest points
    - Lower power values (<2) create smoother surfaces with more distant points having more influence
    - IDW is an exact interpolator - estimated values at sample points equal their measured values
    
    Args:
        input_points: The path to the input point feature class.
                     Must contain points with the values to interpolate.
        z_field: The field containing the values to interpolate.
                Must be a numeric field in the input feature class.
        output_raster: The path to save the output interpolated raster.
                      Will be overwritten if it already exists.
        cell_size: The cell size for the output raster in the same units as the spatial reference.
                  Smaller values create higher resolution outputs but increase processing time.
        power: The power parameter that controls the significance of surrounding points.
              Default is 2. Range is typically 0.5 to 3.
              Higher values give more influence to closer points.
    
    Returns:
        A message indicating success or failure with details about the operation.
        
    Example:
        >>> idw_interpolation("rainfall_stations.shp", "annual_mm", "rainfall_surface.tif", 100, 2)
        "Successfully created IDW interpolation raster: 'rainfall_surface.tif'."
    
    Notes:
        - Requires Spatial Analyst extension
        - Best suited for datasets where closer points are more related than distant ones
        - Creates a smooth surface except at input points
        - Not recommended for data with significant spatial trends or barriers
        - Interpolated values are limited to the range of input values (no extrapolation)
        - Consider using cross-validation to determine the optimal power value
    """
    try:
        # Validate inputs
        if not _feature_class_exists(input_points):
            return f"Error: Input feature class '{input_points}' does not exist."
        
        # Check if the input is a point feature class
        desc = arcpy.Describe(input_points)
        if desc.shapeType != "Point":
            return f"Error: Input feature class must be of point type. Current type: {desc.shapeType}"
        
        if not _is_valid_field_name(input_points, z_field):
            return f"Error: Field '{z_field}' does not exist in '{input_points}'."
        
        if not isinstance(cell_size, (int, float)) or cell_size <= 0:
            return "Error: cell_size must be a positive number."
        
        if not isinstance(power, (int, float)) or power <= 0:
            return "Error: power must be a positive number."
        
        # Set up the IDW parameters
        idw_params = arcpy.sa.RadiusVariable(12, 15000)
        
        # Execute IDW
        arcpy.CheckOutExtension("Spatial")
        try:
            result_raster = arcpy.sa.Idw(
                input_points, 
                z_field, 
                cell_size, 
                power, 
                idw_params)
            
            result_raster.save(output_raster)
            arcpy.CheckInExtension("Spatial")
            return f"Successfully created IDW interpolation raster: '{output_raster}'."
        except Exception as e:
            return f"Error in IDW interpolation: {str(e)}"
        finally:
            arcpy.CheckInExtension("Spatial")
    except arcpy.ExecuteError:
        return f"ArcPy error: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def kriging_interpolation(input_points: str, z_field: str, output_raster: str,
                         cell_size: float, kriging_model: str = "SPHERICAL") -> str:
    """Performs Kriging interpolation on point data to create a continuous surface.
    
    Kriging is a geostatistical interpolation technique that uses statistical models to predict 
    values at unsampled locations based on the spatial autocorrelation of measured points. 
    Unlike simpler methods, Kriging provides both prediction values and measures of uncertainty.
    
    GIS Concepts:
    - Kriging assumes spatial autocorrelation exists in the data
    - It uses semivariograms to model spatial dependence between points
    - Different semivariogram models (spherical, exponential, etc.) fit different spatial patterns
    - Kriging provides the best linear unbiased prediction at unsampled locations
    - It can account for directional influences (anisotropy) in the data
    - Provides standard error maps to assess prediction uncertainty
    
    Args:
        input_points: The path to the input point feature class.
                     Must contain points with the values to interpolate.
        z_field: The field containing the values to interpolate.
                Must be a numeric field in the input feature class.
        output_raster: The path to save the output interpolated raster.
                      Will be overwritten if it already exists.
        cell_size: The cell size for the output raster in the same units as the spatial reference.
                  Smaller values create higher resolution outputs but increase processing time.
        kriging_model: The semivariogram model to use for the interpolation.
                      Options include "SPHERICAL", "CIRCULAR", "EXPONENTIAL", "GAUSSIAN", "LINEAR".
                      Default is "SPHERICAL" which works well for many environmental variables.
    
    Returns:
        A message indicating success or failure with details about the operation.
        
    Example:
        >>> kriging_interpolation("soil_samples.shp", "pH_value", "soil_pH_surface.tif", 50, "EXPONENTIAL")
        "Successfully created Kriging interpolation raster: 'soil_pH_surface.tif'."
    
    Notes:
        - Requires Spatial Analyst extension
        - More computationally intensive than IDW or spline methods
        - Best suited for data with known spatial correlation structure
        - Performs well with irregularly spaced sample points
        - Optimal for environmental variables like precipitation, soil properties, or pollution
        - Consider exploratory spatial data analysis before selecting a semivariogram model
        - For best results, have at least 30-50 well-distributed sample points
    """
    try:
        # Validate inputs
        if not _feature_class_exists(input_points):
            return f"Error: Input feature class '{input_points}' does not exist."
        
        # Check if the input is a point feature class
        desc = arcpy.Describe(input_points)
        if desc.shapeType != "Point":
            return f"Error: Input feature class must be of point type. Current type: {desc.shapeType}"
        
        if not _is_valid_field_name(input_points, z_field):
            return f"Error: Field '{z_field}' does not exist in '{input_points}'."
        
        if not isinstance(cell_size, (int, float)) or cell_size <= 0:
            return "Error: cell_size must be a positive number."
        
        valid_models = ["SPHERICAL", "CIRCULAR", "EXPONENTIAL", "GAUSSIAN", "LINEAR"]
        if kriging_model.upper() not in valid_models:
            return f"Error: Invalid kriging_model. Must be one of: {', '.join(valid_models)}"
        
        # Execute Kriging
        arcpy.CheckOutExtension("Spatial")
        try:
            # Create a KrigingModelOrdinary object
            kriging_params = arcpy.sa.KrigingModelOrdinary(kriging_model.upper())
            
            result_raster = arcpy.sa.Kriging(
                input_points, 
                z_field, 
                kriging_params, 
                cell_size)
            
            result_raster.save(output_raster)
            arcpy.CheckInExtension("Spatial")
            return f"Successfully created Kriging interpolation raster: '{output_raster}'."
        except Exception as e:
            return f"Error in Kriging interpolation: {str(e)}"
        finally:
            arcpy.CheckInExtension("Spatial")
    except arcpy.ExecuteError:
        return f"ArcPy error: {arcpy.GetMessages()}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Update __all__ to include the new tools
__all__ = [
    'add_field','append_features','aspect','batch_process_features','batch_process_rasters',
    'buffer_features','calculate_field','calculate_ndvi','calculate_savi','calculate_tpi',
    'clip_features','closest_facility','create_feature_class','create_file_geodatabase',
    'dataset_exists','define_projection','delete_features','describe_dataset',
    'dissolve_features','download_landsat_tool','erase_features','export_to_csv',
    'extract_by_mask','get_environment_settings','scan_workspace_directory_for_gis_files','hillshade',
    'idw_interpolation','import_csv','intersect_features','kriging_interpolation',
    'list_fields','merge_features','nearest_neighbor_analysis','project_features',
    'raster_calculator','reclassify_raster', 'repair_geometry','route',
    'select_features','service_area','slope','spatial_autocorrelation', 'spatial_join',
    'union_features','zonal_statistics', 'scan_external_directory_for_gis_files'
]