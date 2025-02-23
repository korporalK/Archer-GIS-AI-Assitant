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

# Feature Class Tools
@tool
def buffer_features(input_features: str, output_features: str, buffer_distance: float, buffer_unit: str) -> str:
    """Buffers the input features by the specified distance and unit.

    Args:
        input_features: The path to the input feature class.
        output_features: The path to the output feature class.
        buffer_distance: The buffer distance (e.g., 100).
        buffer_unit: The buffer unit (e.g., 'Meters', 'Feet', 'Kilometers').

    Returns:
        A message indicating success or failure.
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
def get_workspace_inventory(workspace: str = None) -> str:
    """Lists all datasets, feature classes, and tables in the current or specified workspace with detailed information.
    
    This tool provides a comprehensive inventory of the ArcGIS workspace contents, including:
    - Datasets with their types and counts
    - Feature classes with their geometry types, spatial references, and feature counts
    - Tables with their row counts
    
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


# Add more tools following the same pattern... 

# At the bottom of tools.py
__all__ = [
    'add_field',
    'append_features',
    'aspect',
    'buffer_features',
    'calculate_field',
    'clip_features',
    'closest_facility',
    'create_feature_class',
    'create_file_geodatabase',
    'dataset_exists',
    'define_projection',
    'delete_features',
    'describe_dataset',
    'dissolve_features',
    'download_landsat_tool',
    'erase_features',
    'export_to_csv',
    'extract_by_mask',
    'get_environment_settings',
    'get_workspace_inventory',
    'hillshade',
    'import_csv',
    'intersect_features',
    'list_fields',
    'merge_features',
    'project_features',
    'reclassify_raster',
    'repair_geometry',
    'route',
    'select_features',
    'service_area',
    'slope',
    'spatial_join',
    'union_features',
    'zonal_statistics'
]