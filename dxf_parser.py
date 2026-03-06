# import ezdxf
# import matplotlib.pyplot as plt
# from ezdxf.addons.drawing import RenderContext, Frontend
# from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# def clean_and_plot_dxf(file_path, exclude_layers):
#     print(f"[+] Loading DXF: {file_path}")
#     try:
#         doc = ezdxf.readfile(file_path)
#         msp = doc.modelspace()
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         return

#     # 1. REMOVE UNWANTED LAYERS
#     print(f"[+] Removing entities on layers: {exclude_layers}")
#     deleted_count = 0
    
#     # We collect entities first, then delete them to avoid modifying a list while iterating
#     entities_to_delete = []
#     for entity in msp:
#         # Check if the entity's layer matches our exclude list
#         if entity.dxf.layer in exclude_layers:
#             entities_to_delete.append(entity)
            
#     # Delete them from the modelspace
#     for entity in entities_to_delete:
#         msp.delete_entity(entity)
#         deleted_count += 1
        
#     print(f"[i] Deleted {deleted_count} entities.")

#     # (Optional) Save the cleaned DXF file so you can open it in AutoCAD later
#     clean_filename = file_path.replace(".dxf", "_CLEANED.dxf")
#     doc.saveas(clean_filename)
#     print(f"[i] Saved cleaned DXF to: {clean_filename}")

#     # 2. PLOT THE REMAINING DXF
#     print("[+] Generating Matplotlib Plot...")
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     # Setup ezdxf rendering context
#     ctx = RenderContext(doc)
#     out = MatplotlibBackend(ax)
#     frontend = Frontend(ctx, out)
    
#     # Draw the modelspace layout
#     frontend.draw_layout(msp, finalize=True)
    
#     # Formatting the plot
#     plt.title(f"Floor Plan (Excluded: {', '.join(exclude_layers)})", fontsize=14, fontweight='bold')
#     plt.axis('equal') # Keep architectural proportions accurate
    
#     print("[i] Opening plot window...")
#     plt.show()

# if __name__ == "__main__":
#     # Put the path to your DXF file here
#     DXF_FILE = r"C:\Users\abine\OneDrive\Documents\Icebergs_sudharsan\CAD_R&D\Graph Based CAD\DXF_FIles\House Floor Plan Sample.dxf"
    
#     # The layers you want to delete
#     LAYERS_TO_HIDE = ["Furniture", "plants", "Planters", "Stairs"]
    
#     clean_and_plot_dxf(DXF_FILE, LAYERS_TO_HIDE)







import ezdxf

def remove_layers_and_save(input_filepath, output_filepath, exclude_layers):
    print(f"Loading {input_filepath}...")
    doc = ezdxf.readfile(input_filepath)
    msp = doc.modelspace()

    # Find all entities that belong to the layers we want to hide
    entities_to_delete = []
    for entity in msp:
        if entity.dxf.layer in exclude_layers:
            entities_to_delete.append(entity)

    # Delete them from the document
    for entity in entities_to_delete:
        msp.delete_entity(entity)

    # Save the modified document as a new DXF file
    doc.saveas(output_filepath)
    print(f"Success! Cleaned DXF saved as: {output_filepath}")

if __name__ == "__main__":
    DXF_FILE = r"C:\Users\Admin\Desktop\files_folder_ML\hvac\converted_dxf\SEATING ARRANGMENT Sample 1.dxf"
    NEW_DXF_FILE = r"C:\Users\Admin\Desktop\files_folder_ML\hvac\converted_dxf\SEATING ARRANGMENT Sample 1_CLEANED.dxf"
    
    LAYERS_TO_HIDE = ["Edges","H","ceco.net 142","ceco.net 1","A","Block","DIM","DIMLINE","ceco.net 343","ceco.net 345","ceco.net 367","ceco.net 360","TEXT-1","chair","TEXT","SEC", "Furniture", "plants", "Planters"]
    
    remove_layers_and_save(DXF_FILE, NEW_DXF_FILE, LAYERS_TO_HIDE)