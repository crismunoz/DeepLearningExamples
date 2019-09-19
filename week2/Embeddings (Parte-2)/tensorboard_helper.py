# -*- coding: utf-8 -*-
import csv

def CreateTensorboardData(tensor_filename, vectors, metadatos, colnames=None):

    out_file_tsv      = tensor_filename + '_tensor.tsv'
    out_file_tsv_meta = tensor_filename + '_metadata.tsv'
    
    with open(out_file_tsv, 'w',encoding='utf-8') as f:   
        for vector in vectors:
            vector_str = "\t".join([str(x) for x in vector])
            f.write( vector_str + '\n')

    with open(out_file_tsv_meta, 'w',encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        if len(metadatos)>=2:
            if colnames is None:
                colnames = "\t".join([str(i) for i in range(len(metadatos))])
            writer.writerow(colnames)
            for metadato in zip(*metadatos):
                line = [str(x) for x in metadato]
                writer.writerow(line)
        else:
            for metadato in metadatos[0]:
                writer.writerow([metadato])
            
    print("Arquivo com o Tensor 2D foi salvado em: %s" % out_file_tsv)
    print("Arquivo com o Tensor de metadatos foi salvado em: %s" % out_file_tsv_meta)