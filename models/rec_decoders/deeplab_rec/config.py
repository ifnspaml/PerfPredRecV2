layer_map = {
    "seg_decoder": "loaded_model.backbone.decoder", 
    "seg_spp"    : "loaded_model.backbone.aspp", 
    "rec_decoder": "loaded_model.rec_decoder", 
    "rec_spp"    : "loaded_model.aspp"
}

""" Freezing routes """
decoder_layers = {
    'bw':{    
        0: None,
        1: ['last_conv.1'],
        2: ['last_conv.1', 'last_conv.0'],
        3: ['last_conv.1', 'last_conv.0', 'aspp'],
        4: ['last_conv.1', 'last_conv.0', 'aspp', 'conv1', 'bn1'],
        'propagation': 'Freezing propagates backwards.'
    },
    'new_bw':{    
        0: None,
        1: ['last_conv.1'],
        2: ['last_conv.1', 'last_conv.0'],
        3: ['last_conv.1', 'last_conv.0', 'conv1', 'bn1'],
        4: ['last_conv.1', 'last_conv.0', 'conv1', 'bn1', 'aspp'],
        'propagation': 'Freezing propagates backwards.'
    },
    'lat_bw':{    
        0: None,
        1: ['aspp'],
        2: ['aspp', 'last_conv.1'],
        3: ['aspp', 'last_conv.1', 'last_conv.0'],
        4: ['aspp', 'last_conv.1', 'last_conv.0', 'conv1', 'bn1'],
        'propagation': 'Lateral backward freezing.'
    },
}