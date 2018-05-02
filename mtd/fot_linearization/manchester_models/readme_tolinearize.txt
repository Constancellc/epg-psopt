To linearise these files we need to:

1. Create a master dss file, adding

Clear
New object=circuit.manc ! or some other name

2. Remove loadshapes, monitors. Go through and create loads file without a load shape (we call it 'load_snap')

3. Increase the source impedance (if required)

4. Move Transformers to immediately after source changes

5. Copy to create a 'master_y' file, transformers_y file.

NB you can probably use the master_template files attached (the loads_snap and transformers_y will probably have to be created)