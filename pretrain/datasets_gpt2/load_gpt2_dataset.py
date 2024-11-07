def main():
    import glob

    data_dir = '/data/data/llm_data/raw_data_gpt2'
    # data_files = glob.glob(f'{data_dir}/*.jsonl')

    from datasets import load_dataset
    load_dataset(
        path=data_dir,
        # extension='jsonl',
        # data_files=data_files,
        cache_dir='/data/data/cache',
        # **dataset_args,
    )


if __name__ == '__main__':
    main()
