from Dataset import get_dataset, get_args

if __name__ == "__main__":
    args = get_args()

    ds = get_dataset(args=args)

    print(ds[100]["translation"])