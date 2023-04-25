layer_map = {
    "seg_decoder": "loaded_model.backbone.seg.decoder", 
    "rec_decoder": "loaded_model.rec.decoder", 
}

""" Freezing routes """
decoder_layers = {
    'bw':{    
        0: None,
        1: ['step_9'],
        2: ['step_9', 'step_8'],
        3: ['step_9', 'step_8', 'step_7'],
        4: ['step_9', 'step_8', 'step_7', 'step_6'],
        5: ['step_9', 'step_8', 'step_7', 'step_6', 'step_5'],
        6: ['step_9', 'step_8', 'step_7', 'step_6', 'step_5', 'step_4'],
        7: ['step_9', 'step_8', 'step_7', 'step_6', 'step_5', 'step_4', 'step_3'],
        8: ['step_9', 'step_8', 'step_7', 'step_6', 'step_5', 'step_4', 'step_3', 'step_2'],
        9: ['step_9', 'step_8', 'step_7', 'step_6', 'step_5', 'step_4', 'step_3', 'step_2', 'step_1'],
        10: ['step_9', 'step_8', 'step_7', 'step_6', 'step_5', 'step_4', 'step_3', 'step_2', 'step_1', 'step_0'],
        'propagation': 'Freezing propagates backwards to encoder.'
    },
}