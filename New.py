import aspose.slides as slides

def compare_ppts(ppt1_path, ppt2_path, output_path):
    # Load presentations
    pres1 = slides.Presentation(ppt1_path)
    pres2 = slides.Presentation(ppt2_path)
    
    # Ensure PPT2 has at least as many slides as PPT1
    while pres2.slides.count < pres1.slides.count:
        pres2.slides.add_clone(pres1.slides[pres2.slides.count])
    
    for i in range(pres1.slides.count):
        slide1 = pres1.slides[i]
        slide2 = pres2.slides[i]
        
        # Compare shapes (text, tables, images)
        compare_shapes(slide1, slide2)
    
    # Save modified PPT2
    pres2.save(output_path, slides.export.SaveFormat.PPTX)

def compare_shapes(slide1, slide2):
    # Track matched shapes to avoid duplicates
    matched_shapes = set()
    
    # Check for missing shapes in slide2 (present in slide1)
    for shape1 in slide1.shapes:
        found = False
        for shape2 in slide2.shapes:
            if shape1.__class__ == shape2.__class__:
                if isinstance(shape1, slides.AutoShape) and isinstance(shape2, slides.AutoShape):
                    if shape1.text_frame.text == shape2.text_frame.text:
                        found = True
                        matched_shapes.add(shape2)
                        break
                elif isinstance(shape1, slides.Table) and isinstance(shape2, slides.Table):
                    if compare_tables(shape1, shape2):
                        found = True
                        matched_shapes.add(shape2)
                        break
                elif isinstance(shape1, slides.PictureFrame) and isinstance(shape2, slides.PictureFrame):
                    if shape1.name == shape2.name:  # Compare alt-text/name
                        found = True
                        matched_shapes.add(shape2)
                        break
        
        # If shape1 not found in slide2, add it (in blue)
        if not found:
            new_shape = slide2.shapes.add_clone(shape1)
            if isinstance(new_shape, slides.AutoShape):
                for paragraph in new_shape.text_frame.paragraphs:
                    for portion in paragraph.portions:
                        portion.portion_format.fill_format.fill_type = slides.FillType.SOLID
                        portion.portion_format.fill_format.solid_fill_color.color = slides.Color.BLUE
            elif isinstance(new_shape, slides.Table):
                format_table_as_new(new_shape)
    
    # Highlight extra shapes in slide2 (not in slide1)
    for shape2 in slide2.shapes:
        if shape2 not in matched_shapes:
            highlight_shape(shape2)

def compare_tables(table1, table2):
    if table1.rows.count != table2.rows.count or table1.columns.count != table2.columns.count:
        return False
    for i in range(table1.rows.count):
        for j in range(table1.columns.count):
            cell1 = table1.rows[i][j]
            cell2 = table2.rows[i][j]
            if cell1.text != cell2.text:
                return False
    return True

def highlight_shape(shape):
    if isinstance(shape, slides.AutoShape):
        shape.fill_format.fill_type = slides.FillType.SOLID
        shape.fill_format.solid_fill_color.color = slides.Color.YELLOW
    elif isinstance(shape, slides.Table):
        shape.line_format.fill_format.fill_type = slides.FillType.SOLID
        shape.line_format.fill_format.solid_fill_color.color = slides.Color.RED

# Usage
compare_ppts("presentation1.pptx", "presentation2.pptx", "output.pptx")
