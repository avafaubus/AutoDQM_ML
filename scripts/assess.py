# updated assess.py
# only plots if given arg plots_only

# modified assess to have all anomaly scores w run lumi in txt file
# plots most anomalous
#assess.py                                                                                                                                                                                                            

import os
import json
import argparse
import awkward
import numpy

import pandas

from autodqm_ml.utils import setup_logger
from autodqm_ml.utils import expand_path
from autodqm_ml.plotting.plot_tools import make_original_vs_reconstructed_plot, make_sse_plot, plot_roc_curve, plot_rescaled_score_hist
from autodqm_ml.evaluation.roc_tools import calc_roc_and_unc, print_eff_table
from autodqm_ml.constants import kANOMALOUS, kGOOD

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help = "output directory to place files in",
        type = str,
        required = False,
        default = "output"
    )
    parser.add_argument(
        "--input_file",
        help = "input file (i.e. output from fetch_data.py) to use for training the ML algorithm",
        type = str,
        required = False,
        default = None
    )
    parser.add_argument(
        "--histograms",
        help = "csv list of histograms to assess",
        type = str,
        required = True,
        default = None
    )
    parser.add_argument(
        "--algorithms",
        help = "csv list of algorithm names to assess",
        type = str,
        required = False,
        default = None
    )
    parser.add_argument(
        "--n_samples",
        help = "number of samples to make original/reconstructed plots for",
        type = int,
        required = False,
        default = 3
    )
    parser.add_argument(
        "--samples",
        help = "csv list of samples to make original/reconstructed plots for",
        type = str,
        required = False,
        default = None
    )
    parser.add_argument(
        "--make_webpage",
        required=False,
        action="store_true",
        help="make a nicely browsable web page"
    )
    parser.add_argument(
        "--hist_layout",
        type = str,
        required = False,
        default = 'flatten'
    )
    parser.add_argument(
        "--debug",
        help = "run logger in DEBUG mode (INFO is default)",
        required = False,
        action = "store_true"
    )
    parser.add_argument(
        "--plots_only",
        help = "only make plots, no other assessment stats will be produced",
        required = False,
        action = "store_true"
    )
    parser.add_argument(
        "--dump_hists",
        help = "writes contents of specified samples into a txt file",
        required = False,
        action = "store_true"
    )

    return parser.parse_args()


def infer_algorithms(samples, histograms, algorithms):
    for histogram, info in histograms.items():
        for field in samples.fields:
            if field == histogram:
                histograms[histogram]["original"] = field
            elif histogram in field and "_score_" in field:
                algorithm = field.replace(histogram, "").replace("_score_", "")
                if algorithms is not None:
                    if algorithm not in algorithms:
                        continue

                if not algorithm in info["algorithms"].keys():
                    histograms[histogram]["algorithms"][algorithm] = { "score" : field }

                # Check if a reconstructed histogram also exists for algorithm                                                                                                                                        
                reco = field.replace("score", "reco")
                if reco in samples.fields:
                    histograms[histogram]["algorithms"][algorithm]["reco"] = reco
                else:
                    histograms[histogram]["algorithms"][algorithm]["reco"] = None

    return histograms

def plot_reco_hists(args, stats, histograms, samples, logger):
    # Plots of original/reconstructed histograms
    if args.samples is None:
        random_samples = True
        selected_samples_idx = numpy.random.choice(len(samples), size=args.n_samples, replace=False)
        selected_samples = samples.lumi[selected_samples_idx]
        logger.debug("[assess.py] An explicit list of samples was not given, so we will make plots for %d randomly chosen samples: %s" % (args.n_samples, str(selected_samples)))
    else:
        random_samples = False
        selected_samples_idx = samples.lumi < 0  # dummy all False

        for pair in args.samples.split(","):
            run, lumi = map(int, pair.split(":"))
            selected_samples_idx = selected_samples_idx | (
                (samples.run_number == run) &
                (samples.lumi == lumi)
            )

    logger.debug(
        "[assess.py] Will make plots for the specified run:lumi pairs: %s"
        % args.samples
    )

    mean_histogram_set = {}
    for h, info in histograms.items():
        original_histogram = samples[info["original"]]
        mean_histogram = awkward.mean(original_histogram, axis=0)
        mean_histogram_set[h] = mean_histogram

    samples_trim = samples[selected_samples_idx]
    for h, info in histograms.items():
        stats_checked = False
        for i in range(len(samples_trim)):
            sample = samples_trim[i]
            lumi_number = sample.lumi
            run_number = sample.run_number
            original = sample[info["original"]]
            recos = {}
            for algorithm, algorithm_info in info["algorithms"].items():
                if algorithm_info["reco"] is None:                                                                                                                                                              
                    continue
                recos[algorithm] = { "reco" : sample[algorithm_info["reco"]], "score" : sample[algorithm_info["score"]]}

                if not stats_checked:
                    if sample[algorithm_info["reco"]].ndim > 1:
                        stats['dim_0'].append(len(sample[algorithm_info["reco"]]))
                        stats['dim_1'].append(len(sample[algorithm_info["reco"]][0]))
                    else:
                        stats['dim_0'].append(len(sample[algorithm_info["reco"]]))
                        stats['dim_1'].append(numpy.nan)
            stats_checked = True
            h_name = '_'.join(h.split("/")[3:])
            save_name = args.output_dir + "/" + h_name + "/Run%d" % run_number + "_Lumi%d.pdf" % lumi_number
            make_original_vs_reconstructed_plot(h_name, original, recos, mean_histogram_set[h], run_number, lumi_number, save_name, hist_layout = args.hist_layout)

    logger.info("[assess.py] Plots written to directory '%s'." % (args.output_dir))    


def dump_hists(args, histograms, samples, logger):
    selected_samples_idx = awkward.zeros_like(samples.lumi, dtype=bool)
    
    if args.samples is None:
        logger.error("--dump_hists requires --samples.")
        return

    for pair in args.samples.split(","):
        run, lumi = map(int, pair.split(":"))
        selected_samples_idx = selected_samples_idx | (
            (samples.run_number == run)
            & (samples.lumi == lumi)
        )

    samples_trim = samples[selected_samples_idx]
    logger.debug(
        "[assess.py] Will dump histogram contents for the specified run:lumi pairs: %s"
        % args.samples
    )
    
    for h, info in histograms.items():

        h_name = "_".join(h.split("/")[3:])

        for sample in samples_trim:

            run = sample.run_number
            lumi = sample.lumi

            outfile = os.path.join(
                args.output_dir,
                h_name,
                f"Run{run}_Lumi{lumi}_hist.txt",
            )
            hist = sample[info["original"]]

            with open(outfile, "w") as f:

                f.write(f"Histogram: {h}\n")
                f.write(f"Run: {run}\n")
                f.write(f"Lumi: {lumi}\n")
                f.write("=" * 80 + "\n")

                if hist.ndim == 1:
                    f.write("Bin\tContent\n")
                    for i, value in enumerate(hist):
                        f.write(f"{i}\t{value}\n")

                elif hist.ndim == 2:
                    for iy, row in enumerate(hist):
                        for ix, value in enumerate(row):
                            f.write(f"{ix}\t{iy}\t{value}\n")
    logger.info("[assess.py] Histogram contents written to directory '%s'." % (args.output_dir))    


def main(args):

    os.system("mkdir -p %s/" % args.output_dir)
    
    logger_mode = "DEBUG" if args.debug else "INFO"
    log_file = "%s/fetch_data_log_%s.txt" % (args.output_dir, "assess")
    logger = setup_logger(logger_mode, log_file)
    
    stats = {
             'hist': [],
             'dim_0': [],
             'dim_1': [],
             'algo': [],
             'avg_an_score': [],
             'std_an_score': []
    }
    histograms = { x : {"algorithms" : {}} for x in args.histograms.split(",") }
    
    samples = awkward.from_parquet(args.input_file)
    if args.algorithms is not None:
        algorithms = args.algorithms.split(",")
    else:
        algorithms = None
    
    histograms = infer_algorithms(samples, histograms, algorithms)
    for h, info in histograms.items():
        logger.debug("[assess.py] For histogram '%s', found the following anomaly detection algorithms:" % (h))
        for a, a_info in info["algorithms"].items():
            logger.debug("\t Algorithm '%s' with score in field '%s' and reconstructed histogram in field '%s'" % (a, a_info["score"], str(a_info["reco"])))
    if not args.plots_only:
        # Print out samples with N highest sse scores for each histogram                                                                                                                                                  
        N = 10
        for h, info in histograms.items():
            score_hist_data = {'algo':[], 'score':[], 'bad':[]} #track scores for histogram                                                                                                                               
            for algorithm, algorithm_info in info["algorithms"].items():
                samples_sorted = samples[awkward.argsort(samples[algorithm_info["score"]], ascending=False)]
    
                h_name = "_".join(h.split("/")[3:])
                txt_name = f"{h_name}_{algorithm}_scores.txt"
                txt_path = os.path.join(args.output_dir, txt_name)
                
                with open(txt_path, "w") as f:
                    f.write(f"Histogram: {h}\n")
                    f.write(f"Algorithm: {algorithm}\n")
                    f.write("=" * 80 + "\n")
                    f.write("Rank\tRun\tLumi\tLabel\tAnomaly Score\n")
                    for i in range(len(samples_sorted)):
                        f.write(
                                f"{i+1}\t"
                                f"{samples_sorted.run_number[i]}\t"
                                f"{samples_sorted.lumi[i]}\t"
                                f"{samples_sorted.label[i]}\t"
                                f"{samples_sorted[algorithm_info['score']][i]:.6e}\n")
                
                logger.info("[assess.py] For histogram '%s', algorithm '%s', the mean +/- std anomaly score is: %.2e +/- %.2e." % (h, algorithm, awkward.mean(samples[algorithm_info["score"]]), awkward.std(samples[algorithm_info["score"]])))
                #logger.info("[assess.py] For histogram '%s', algorithm '%s', the runs with the highest anomaly scores are: " % (h, algorithm))                                                                           
                logger.info("\t The runs and lumisections with the highest anomaly scores are:")
                for i in range(N):
                    logger.info("\t Run number : %d, Lumisection : %d, Anomaly Score : %.2e" % (samples_sorted.run_number[i], samples_sorted.lumi[i], samples_sorted[algorithm_info["score"]][i]))
                stats['hist'].append(h)
                stats['algo'].append(algorithm)
                stats['avg_an_score'].append(awkward.mean(samples[algorithm_info["score"]]))
                stats['std_an_score'].append(awkward.std(samples[algorithm_info["score"]]))
    
                if len(numpy.unique(samples['label'])) > 1:
                    score_hist_data['algo'].append(algorithm)
                    score_hist_data['score'].append(samples[algorithm_info["score"]][samples['label'] == 0])
                    score_hist_data['bad'].append(samples[algorithm_info["score"]][samples['label'] == 1])
            h = '_'.join(h.split("/")[3:])
            if not os.path.isdir(args.output_dir + "/" + h + "/"):
                os.mkdir(args.output_dir + "/" + h + "/")
            #if len(score_hist_data['algo']) != 0:                                                                                                                                                                        
                #plot_rescaled_score_hist(score_hist_data, h, args.output_dir + "/" + h + "/" + "score_hist.png")                                                                                                         
        # Histogram of sse for algorithms                                                                                                                                                                                 
        splits = {
                "label" : [("train", 0), ("test", 1)]}
        for h, info in histograms.items():
            for split, split_info in splits.items():
                recos_by_label = { k : {} for k,v in info["algorithms"].items() }
                for name, id in split_info:
                    samples_set = samples[samples[split] == id]
                    if len(samples_set) == 0:
                        logger.warning("[assess.py] For histogram '%s', no runs belong to the set '%s', skipping making a histogram of SSE for this." % (h, name))
                        continue
                    recos = {}
                    for algorithm, algorithm_info in info["algorithms"].items():
                        recos[algorithm] = { "score" : samples_set[algorithm_info["score"]] }
                        recos_by_label[algorithm][name] = { "score" : samples_set[algorithm_info["score"]] }
                    h_name = '_'.join(h.split("/")[3:])
                    save_name = args.output_dir + "/" + h_name + "/sse_%s_%s.pdf" % (split, name)
                    make_sse_plot(h_name, recos, save_name)
    
                for algorithm, recos_alg in recos_by_label.items():
                    if not recos_alg:
                        continue
                    save_name = args.output_dir + "/" + h_name + "/sse_%s_%s.pdf" % (algorithm, split)
                    #make_sse_plot(h_name, recos_alg, save_name)                                                                
        # ROC curves (if there are labeled runs)                                                                                                                                                                          
        has_labeled_samples = {h:True for h in histograms}
        labeled_samples_cut = {h:samples.lumi < 0 for h in histograms}
        for h, info in histograms.items():
            for name, id in splits["label"]:
                cut = samples['label'] == id
                labeled_samples_cut[h] = labeled_samples_cut[h] | cut
                samples_set = samples[cut]
                has_labeled_samples[h] = has_labeled_samples[h] and (len(samples_set) > 0)
        roc_results = {}
        for h, info in histograms.items():
            if has_labeled_samples[h]:
                labeled_samples = samples[labeled_samples_cut[h]]
                roc_results[h] = {}
                for algorithm, algorithm_info in info["algorithms"].items():
                    pred = labeled_samples[algorithm_info["score"]]
                    roc_results[h][algorithm] = calc_roc_and_unc(labeled_samples['label'], pred)
    
                h_name = '_'.join(h.split("/")[3:])
                save_name = args.output_dir + "/" + h_name + "/roc.pdf"
                #plot_roc_curve(h_name, roc_results[h], save_name)                                                                                                                                                        
                #plot_roc_curve(h_name, roc_results[h], save_name.replace(".pdf", "_log.pdf"), log = True)                                                                                                                
                #print_eff_table(h_name, roc_results[h])                                                                                                                                                                  

        stat_parquet_dir = args.output_dir + "/assessment_stats.parquet"
        stat_csv_dir = args.output_dir + "/assessment_stats.csv"
        pandas.DataFrame(stats).to_csv(stat_csv_dir)
        pandas.DataFrame(stats).to_parquet(stat_parquet_dir)
        logger.info("[assess.py] Assessment statistics written to '%s' and '%s'" % (stat_parquet_dir, stat_csv_dir))
        if args.make_webpage:
            os.system("cp web/index.php %s" % args.output_dir)
            os.system("chmod 755 %s" % args.output_dir)
            os.system("chmod 755 %s/*" % args.output_dir)
            
    for h in histograms:
        h_name = "_".join(h.split("/")[3:])
        hist_dir = os.path.join(args.output_dir, h_name)
        os.makedirs(hist_dir, exist_ok=True)
    plot_reco_hists(args, stats, histograms, samples, logger)
    if args.dump_hists:
        dump_hists(args, histograms, samples, logger)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)


