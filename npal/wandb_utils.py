import wandb
import pathlib
import numpy as np


NP_NAME_DICT = {
    "np_acq_func": {
        "mean": "mean",
        "mean_plus_stddev": "mps",
    },
    "np_feature_type": {
        "raw": "raw",
        "classifier": "clp",
    },
}


def wandb_setup(args):
    if args.resume_from_checkpoint:
        # This function assumes the below folder structure:
        #  args.save_dir
        #   - wandb_id.txt
        #   - wandb_dir.txt
        #   models
        #    classifiers
        #     - model.ckpt, potentially args.resume_from_checkpoint
        #    policies
        #     - model.ckpt, potentially args.resume_from_checkpoint

        # Use this to determine which model we're resuming from.
        model_type_dir = pathlib.Path(args.resume_from_checkpoint).parent  # `classifiers` OR `policies`
        # Overwrite out_dir and save_dir with values from checkpoint
        args.save_dir = model_type_dir.parent.parent
        args.out_dir = args.save_dir.parent

        with open(args.save_dir / "wandb_id.txt", "r") as f:
            wandb_id = f.read()
        with open(args.save_dir / "wandb_dir.txt", "r") as f:
            wandb_dir = pathlib.Path(f.read())

    else:
        wandb_id = wandb.util.generate_id()
        wandb_dir = args.save_dir

    wandb.init(
        entity=args.wandb_entity,
        project="{}_{}".format(args.wandb_project, args.data_type),
        config=args,
        resume="allow",
        id=wandb_id,
        dir=str(wandb_dir),  # PosixPath to str
    )

    if not wandb.run.resumed:
        # Extract run index from wandb name
        wandb_index = wandb.run.name.split("-")[-1]
        # Overwrite wandb run name
        wandb_name = make_wandb_run_name(args)
        wandb.run.name = wandb_name + "-" + wandb_index

        # Save wandb info
        with open(args.save_dir / "wandb_name.txt", "w") as f:
            f.write(wandb.run.name)
        with open(args.save_dir / "wandb_id.txt", "w") as f:
            f.write(wandb.run.id)
        with open(args.save_dir / "wandb_dir.txt", "w") as f:
            f.write(str(wandb_dir))

        with open(args.save_dir / str(wandb.run.name), "w") as f:
            f.write(str(wandb.run.name) + "\n" + str(wandb_id))


def make_wandb_run_name(args):
    train_weighting, eval_weighting = "", ""
    if args.class_weight_type:
        eval_weighting = "-{}".format(args.class_weight_type[:3])
        if args.use_class_weights_for_fit:
            train_weighting = "-{}".format(args.class_weight_type[:3])
    acquisition_strategy = args.acquisition_strategy

    if "np_" in acquisition_strategy:
        af = NP_NAME_DICT["np_acq_func"][args.np_acq_func]
        feat = NP_NAME_DICT["np_feature_type"][args.np_feature_type]
        att = ""
        if args.np_attention is not None:
            if args.np_self_attention is not None:
                att = "-satt"
            else:
                att = "-att"
        np_string = "{}-r{}-{}-{}".format(att, args.np_r_dim, af, feat)
        if args.project_np_features:
            np_string += "-prj"
        if args.use_leave_one_out_rewards:
            np_string += "-loo"
        if args.reward_split != "reward":
            np_string += "-{}".format(args.reward_split)
        if args.acquisition_strategy == "np_future":
            np_string += "-nrd{}".format(args.np_num_resampled_datasets)
        np_string += f"-e{args.np_num_epochs}"
        acquisition_strategy += np_string
    elif "hal_" in args.acquisition_strategy:
        acquisition_strategy += "-{}".format(args.hal_exploit_p)
    elif args.acquisition_strategy == "cbal":
        acquisition_strategy += "-l{}".format(args.cbal_lambda)

    model_str = "s{}_{}{}_{}{}".format(
        args.seed, args.classifier_type, train_weighting, acquisition_strategy, eval_weighting
    )

    imbalance_factors = np.array(args.imbalance_factors)
    with_zero = ""
    if 0 in args.imbalance_factors:
        imbalance_factors = imbalance_factors[imbalance_factors != 0]
        with_zero = "w0"
    imba_ratio = max(imbalance_factors) / min(imbalance_factors)

    ignore_existing_imbalance = "ig" if args.ignore_existing_imbalance else ""

    if args.total_num_points is not None:
        total_num_points = args.total_num_points
    else:
        total_num_points = "notot"

    data_str = "ds{}_imb{}{}{}_{}".format(
        args.data_split_seed,
        imba_ratio,  # Better than sending the entire list
        with_zero,
        ignore_existing_imbalance,
        total_num_points,
    )

    al_str = "b{}".format(args.budget)

    return "{}_{}_{}".format(model_str, data_str, al_str)
