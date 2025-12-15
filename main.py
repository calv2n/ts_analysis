import utils
import config

if __name__ == "__main__":
    results_rolling = utils.find_best_model_rolling(
        dataset=config.DATASET, 
        target=config.TARGET, 
        p_candidates=config.P_CANDIDATES, 
        prop_test=config.TEST_PROP
    )
    utils.plot_best_rolling(results_rolling)

    results_uptodate = utils.find_best_model_uptodate(
        dataset=config.DATASET, 
        target=config.TARGET, 
        p_candidates=config.P_CANDIDATES, 
        prop_test=config.TEST_PROP
    )
    utils.plot_best_uptodate(results_uptodate)
