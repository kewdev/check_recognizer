import compress_fasttext

if __name__ == '__main__':
    small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        './../ft_freqprune_100K_20K_pq_100.bin'
    )
    print(len(small_model['!@#$%^&*()']))
