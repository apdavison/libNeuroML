# encoding: utf-8
from __future__ import division
import os
import io
from math import sqrt, pi
import neuroml


class NeuroMLWriter(object):
    @classmethod
    def write(cls,nmldoc,file,close=True):
        """
        Writes from NeuroMLDocument to nml file
        in future can implement from other types
        via chain of responsibility pattern.
        """

        if isinstance(file,str):
            file = open(file,'w')

        #TODO: this should be extracted from the schema:
        namespacedef = 'xmlns="http://www.neuroml.org/schema/neuroml2" '
        namespacedef += ' xmlns:xs="http://www.w3.org/2001/XMLSchema"'
        namespacedef += ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        namespacedef += ' xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2 https://raw.github.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_%s.xsd"'%neuroml.current_neuroml_version

        nmldoc.export(file,0,name_="neuroml",
                      namespacedef_=namespacedef) #name_ param to ensure root element named correctly - generateDS limitation
                      
        if close:
            file.close()


class NeuroMLHdf5Writer(object):
     
    @classmethod
    def write(cls,nml_doc,h5_file_name,embed_xml=True,compress=True):
        
        import tables
        FILTERS = tables.Filters(complib='zlib', complevel=5) if compress else tables.Filters()
        h5file = tables.open_file(h5_file_name, mode = "w", title = nml_doc.id, filters=FILTERS)
        
        rootGroup = h5file.create_group("/", 'neuroml', 'Root NeuroML group')
        
        rootGroup._f_setattr("id", nml_doc.id)
        rootGroup._f_setattr("notes", nml_doc.notes)
        rootGroup._f_setattr("GENERATED_BY", "libNeuroML v%s"%(neuroml.__version__))
        
        for network in nml_doc.networks:

            network.exportHdf5(h5file, rootGroup)
            
        if embed_xml:
            
            networks = []
            for n in nml_doc.networks:
                networks.append(n)
            
            nml_doc.networks = []
            
            try:
                import StringIO
                sf = StringIO.StringIO()
            except:
                import io
                sf = io.StringIO()
            
            NeuroMLWriter.write(nml_doc,sf,close=False)
            
            nml2 = sf.getvalue()
            
            rootGroup._f_setattr("neuroml_top_level", nml2)
            

            # Put back into previous form...
            for n in networks:
                nml_doc.networks.append(n)
            
        h5file.close()  # Close (and flush) the file
        
    '''
    @classmethod
    def write_xml_and_hdf5(cls,nml_doc0,xml_file_name,h5_file_name):
        
        nml_doc_hdf5 = neuroml.NeuroMLDocument(nml_doc0.id)
        
        for n in nml_doc0.networks:
            nml_doc_hdf5.networks.append(n)
            
        nml_doc0.networks = []
        
        nml_doc0.includes.append(neuroml.IncludeType(h5_file_name)) 
        
        NeuroMLWriter.write(nml_doc0,xml_file_name)
        
        NeuroMLHdf5Writer.write(nml_doc_hdf5,h5_file_name,embed_xml=False)
        
        # Put back into previous form...
        for n in nml_doc_hdf5.networks:
            nml_doc0.networks.append(n)
        for inc in nml_doc0.includes:
            if inc.href == h5_file_name:
                nml_doc0.includes.remove(inc)'''
        
        
        
class JSONWriter(object):
    """
    Write a NeuroMLDocument to JSON, particularly useful
    when dealing with lots of ArrayMorphs.
    """

    
    @classmethod
    def __encode_as_json(cls,neuroml_document):
        neuroml_document = cls.__sanitize_doc(neuroml_document)
        from jsonpickle import encode as json_encode
        encoded = json_encode(neuroml_document)
        return encoded
    
    @classmethod
    def __sanitize_doc(cls,neuroml_document):
        """
        Some operations will need to be performed
        before the document is JSON-pickleable.
        """

        for cell in neuroml_document.cells:
            try:
                cell.morphology.vertices = cell.morphology.vertices.tolist()
                cell.morphology.physical_mask = cell.morphology.physical_mask.tolist()
                cell.morphology.connectivity = cell.morphology.connectivity.tolist()
            except:
                pass

        return neuroml_document

    @classmethod
    def __file_handle(file):
        if isinstance(cls,file,str):
            import tables
            fileh = tables.open_file(filepath, mode = "w")

            
    @classmethod    
    def write(cls,neuroml_document,file):
        if isinstance(file,str):
            fileh = open(file, mode = 'w')
        else:
            fileh = file

        if isinstance(neuroml_document,neuroml.NeuroMLDocument):
            encoded = cls.__encode_as_json(neuroml_document)

        else:
            raise NotImplementedError("Currently you can only serialize NeuroMLDocument type in JSON format")

        fileh.write(encoded)
        fileh.close()

    @classmethod
    def write_to_mongodb(cls,neuroml_document,db,host=None,port=None,id=None):
        from pymongo import MongoClient
        import json

        if id == None:
            id = neuroml_document.id
        
        if host == None:
            host = 'localhost'
        if port == None:
            port = 27017

        client = MongoClient(host, port)
        db = client[db]
        collection = db[id]

        if isinstance(neuroml_document,neuroml.NeuroMLDocument):
            encoded = cls.__encode_as_json(neuroml_document)

        encoded_dict = json.loads(encoded)
        collection.insert(encoded_dict)


class ArrayMorphWriter(object):
    """
    For now just testing a simple method which can write a morphology, not a NeuroMLDocument.
    """

    @classmethod
    def __write_single_cell(cls,array_morph,fileh,cell_id=None):
        vertices = array_morph.vertices
        connectivity = array_morph.connectivity
        physical_mask = array_morph.physical_mask

        # Get the HDF5 root group
        root = fileh.root
        
        # Create the groups:
        # can use morphology name in future?

        if array_morph.id == None:
            morphology_name = 'Morphology'
        else:
            morphology_name = array_morph.id

        if cell_id == None:
            morphology_group = fileh.create_group(root, morphology_name)
            hierarchy_prefix = "/" + morphology_name
        else:
            cell_group = fileh.create_group(root, cell_id)
            morphology_group = fileh.create_group(cell_group, morphology_name)
            hierarchy_prefix = '/' + cell_id + '/' + morphology_name

        vertices_array = fileh.create_array(hierarchy_prefix, "vertices", vertices)
        connectivity_array = fileh.create_array(hierarchy_prefix, "connectivity", connectivity)
        physical_mask_array = fileh.create_array(hierarchy_prefix, "physical_mask", physical_mask)

    @classmethod
    def __write_neuroml_document(cls,document,fileh):
        document_id = document.id

        for default_id,cell in enumerate(document.cells):
            morphology = cell.morphology

            if morphology.id == None:
                morphology.id = 'Morphology' + str(default_id)
            if cell.id == None:
                cell.id = 'Cell' + str(default_id)

            cls.__write_single_cell(morphology,fileh,cell_id=cell.id)

        for default_id,morphology in enumerate(document.morphology):

            if morphology.id == None:
                morphology.id = 'Morphology' + str(default_id)

            cls.__write_single_cell(morphology,fileh,cell_id=cell.id)


    @classmethod
    def write(cls,data,filepath):

        import tables
        fileh = tables.open_file(filepath, mode = "w")
        
        #Now instead we should go through a document/cell/morphology
        #hierarchy - this kind of tree traversal should be done recursively

        if isinstance(data,neuroml.arraymorph.ArrayMorphology):
            cls.__write_single_cell(data, fileh)

        if isinstance(data,neuroml.NeuroMLDocument):
            cls.__write_neuroml_document(data,fileh)
            
        # Finally, close the file (this also will flush all the remaining buffers!)
        fileh.close()


class ABCWriter(object):
    """
    docstring needed
    """

    @classmethod
    def write(cls, neuroml_document, output_path):
        """docstring needed"""

        from string import Template
        import numpy as np
        import h5py

        if len(neuroml_document.networks) > 1:
            raise Exception("This format cannot handle multiple Networks in a single document")
        if len(neuroml_document.networks) == 0:
            raise Exception("The document must contain at least one Network")

        if neuroml_document.includes:
            raise NotImplementedError("Unresolved includes."
                                      "Please load the document using `read_neuroml2_file(filename, include_includes=True)`")

        # --- define directory layout ---
        config = {
            "target_simulator": "NEURON",
            "manifest": {
                "$BASE_DIR": "${configdir}",
                "$NETWORK_DIR": "$BASE_DIR/networks",
                "$COMPONENT_DIR": "$BASE_DIR/components"
            },
            "components": {
                "morphologies": "$COMPONENT_DIR/morphologies",
                "synaptic_models": "$COMPONENT_DIR/synapse_dynamics",
                "point_neuron_models": "$COMPONENT_DIR/point_neuron_dynamics",
                "mechanisms":"$COMPONENT_DIR/mechanisms",
                "biophysical_neuron_models": "$COMPONENT_DIR/biophysical_neuron_dynamics",
                "templates": "$COMPONENT_DIR/hoc_templates",
            },
            "networks": {
                "node_files": [
                    {
                        "nodes": "$NETWORK_DIR/nodes.h5",
                        "node_types": "$NETWORK_DIR/node_types.csv"
                    }
                ],
                "edge_files":[
                    {
                        "edges": "$NETWORK_DIR/edges.h5",
                        "edge_types": "$NETWORK_DIR/edge_types.csv"
                    },
                ]
            }
        }

        base_dir = Template(config["manifest"]["$BASE_DIR"]).substitute(configdir=output_path)
        network_dir = Template(config["manifest"]["$NETWORK_DIR"]).substitute(BASE_DIR=base_dir)
        component_dir = Template(config["manifest"]["$COMPONENT_DIR"]).substitute(BASE_DIR=base_dir)

        for directory in (base_dir, network_dir, component_dir):
            os.makedirs(directory)

        # --- export morphologies ---
        morph_path = Template(config["components"]["morphologies"]).substitute(COMPONENT_DIR=component_dir)
        os.mkdir(morph_path)
        for cell in neuroml_document.cells:
            print cell.id
            morphology_to_swc(cell.morphology, 
                              filename=os.path.join(morph_path, cell.id + ".swc"))

        # --- export mechanisms ---
        mech_path = Template(config["components"]["mechanisms"]).substitute(COMPONENT_DIR=component_dir)
        os.mkdir(mech_path)
        

        # --- export nodes ---
        nodes_path = Template(config["networks"]["node_files"][0]["nodes"]).substitute(NETWORK_DIR=network_dir)
        nodes_file = h5py.File(nodes_path, 'w-')  # fail if exists

        #   add some annotations
        nodes_file["id"] = neuroml_document.id
        nodes_file["notes"] = neuroml_document.notes
        if hasattr(neuroml_document, "temperature"):
            nodes_file["temperature"] = neuroml_document.temperature  # I'm not sure this is the right place to put this

        net = neuroml_document.networks[0]

        n = sum(p.get_size() for p in net.populations)
        root = nodes_file.create_group("nodes")  # todo: add attribute with network name
        root.create_dataset("node_gid", shape=(n,), dtype='i4')
        root.create_dataset("node_type_id", shape=(n,), dtype='i2')
        root.create_dataset("node_group", shape=(n,), dtype='S32')  # todo: calculate the max label size
        root.create_dataset("node_group_index", shape=(n,), dtype='i2')

        offset = 0
        for i, population in enumerate(net.populations):
            if population.type != "populationList":
                raise NotImplementedError
            m = population.get_size()
            index = slice(offset, offset + m)
            root["node_gid"][index] = np.arange(offset, offset + m, dtype=int)
            root["node_type_id"][index] = i * np.ones((m,))
            root["node_group"][index] = np.array([population.id] * m)
            root["node_group_index"][index] = np.arange(m, dtype=int)

            node_group = root.create_group(population.id)

            x, y, z = np.array([(i.location.x, i.location.y, i.location.z)
                                for i in population.instances]).T
            node_group.create_dataset('x', data=x)
            node_group.create_dataset('y', data=y)
            node_group.create_dataset('z', data=z)
            #population.component

            offset += m

             
        for syn_conn in net.synaptic_connections:
            pass

        for exp_inp in net.explicit_inputs:
            pass

        for proj in net.projections:
            pass
            
        for eproj in net.electrical_projections:
            pass
            
        for cproj in net.continuous_projections:
            pass
            
        for il in net.input_lists:
            pass

        nodes_file.close()


def morphology_to_swc(morphology, filename=None):
    """
    Export a NeuroML morphology as a standardized SWC file
    """

    # ---- Try to find segment groups corresponding to the SWC "types"
    #      ("soma", "axon", "basal dendrite", "apical dendrite")
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    DENDRITE = 33

    neurolex_ids = {
        "GO:0043025": SOMA,
        "GO:0030424": AXON,
        "GO:0030425": DENDRITE,
        "GO:0097441": BASAL_DENDRITE,
        "GO:0097440": APICAL_DENDRITE
    }
    segment_group_index = {}
    for segment_group in morphology.segment_groups:
        segment_group_index[segment_group.id] = segment_group

    region_index = {}
    for segment_group in morphology.segment_groups:
        if segment_group.neuro_lex_id in neurolex_ids:
            region = neurolex_ids[segment_group.neuro_lex_id]
            region_index[region] = []
            for member in segment_group.members:
                region_index[region].append(member.segment)
            for include in segment_group.includes:
                included_group = segment_group_index[include.segment_groups]  # note plural attibute name but singular value
                for member in included_group.members:
                    region_index[region].append(member.segments)             # note plural attibute name but singular value
                assert len(included_group.includes) == 0  # todo: recursively resolve includes
    
    if SOMA not in region_index:
        raise Exception("Cannot convert to SWC, no soma indicated")
    if AXON not in region_index:
        raise Exception("Cannot convert to SWC, no axon indicated")
    if BASAL_DENDRITE not in region_index:
        if DENDRITE not in region_index:
            raise Exception("Cannot convert to SWC, no dendrites indicated")
        # to be improved. Could also use segment group names to infer regions
        region_index[BASAL_DENDRITE] = region_index.pop(DENDRITE)
        print("Warning: assigning all dendrites as basal dendrites")

    reverse_region_index = {}
    for region, seg_ids in region_index.items():
        for seg_id in seg_ids:
            assert seg_id not in reverse_region_index
            reverse_region_index[seg_id] = region
    # check that all segments have a region
    assert len(reverse_region_index) == len(morphology.segments)

    # ---- Transform non-spherical soma (e.g. cylindrical) to a spherical equivalent
    soma_segments = [seg for seg in morphology.segments if seg.id in region_index[SOMA]]
    if len(soma_segments) == 1:
        seg = soma_segments[0]
        x = (seg.proximal.x + seg.distal.x)/2
        y = (seg.proximal.y + seg.distal.y)/2
        z = (seg.proximal.z + seg.distal.z)/2
        area = frustum_surface(seg.proximal, seg.distal, sides="lateral")
    elif len(soma_segments) == 2:
        seg0, seg1 = soma_segments
        assert seg1.parent.segments == seg0.id
        x = (seg0.proximal.x + seg1.distal.x)/2
        y = (seg0.proximal.y + seg1.distal.y)/2
        z = (seg0.proximal.z + seg1.distal.z)/2
        area = frustum_surface(seg0.proximal, seg0.distal, sides="lateral")
        area += frustum_surface(seg0.distal, seg1.distal, sides="lateral")
    else:
        raise NotImplementedError("todo")
    
    NO_PARENT = -1  # or zero? Seems not to be defined in the spec
    soma_point = {
        "id": 0,
        "type": SOMA,
        "x": x,
        "y": y,
        "z": z,
        "radius": sqrt(area / (4 * pi)),
        "parent": NO_PARENT 
    }

    # ---- Extract points from all segments
    points = [soma_point]
    for segment in morphology.segments:
        if segment.id not in region_index[SOMA]:
            assert segment.parent is not None
            if segment.parent.segments in region_index[SOMA]:  # note plural attribute name but singular value
                parent = soma_point["id"]
            else:
                parent = segment.parent.segments
            # todo: check that segment.proximal (if not None) matches parent.distal
            point = {
                "id": segment.id,
                "type": reverse_region_index[segment.id],
                "x": segment.distal.x,
                "y": segment.distal.y,
                "z": segment.distal.z,
                "radius": segment.distal.diameter / 2,
                "parent": parent
            }
            points.append(point)
    
    # sort the points by NeuroML ID. This is not essential, but is likely to make it easier to check the SWC
    points.sort(key=lambda point: point["id"])

    # construct a map from NeuroML segment id to point index
    id_map = {}
    for i, point in enumerate(points, start=1):
        id_map[point["id"]] = i

    # remap id and parent
    for point in points:
        point["id"] = id_map[point["id"]]
        if point["parent"] != NO_PARENT:
            point["parent"] = id_map[point["parent"]]

    # ---- Now export to swc
    if filename is None:
        filename = morphology.id + ".swc"
    assert filename.endswith(".swc")
    with io.open(filename, "w", encoding="ascii") as fp:
        fp.write(u"# Generated from NeuroML morphology {} by libNeuroML\n".format(morphology.id))
        for point in points:
            fp.write(u"{id} {type} {x} {y} {z} {radius} {parent}\n".format(**point))


def frustum_surface(base, top, sides="all"):
    """
    Returns the surface area of the right conical frustum
    represented by two circles, "base" and "top".

    You can choose to return the partial surface area
    of only the top, the base or the lateral sides.
    """
    possible_sides = ("lateral", "base", "top", "all")
    if sides not in possible_sides:
        raise ValueError("sides must be one of {}".format(possible_sides))
    R = base.diameter / 2
    r = top.diameter / 2
    h = sqrt((top.x - base.x)**2 + (top.y - base.y)**2 + (top.z - base.z)**2)
    A = 0
    if sides in ("lateral", "all"):
        A += pi * (R + r) * sqrt(h**2 + (R - r)**2)
    if sides in ("base", "all"):
        A += pi * R * R
    if sides in  ("top", "all"):
        A += pi * r * r
    return A

