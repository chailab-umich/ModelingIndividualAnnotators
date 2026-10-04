import numpy as np
from tqdm import tqdm
from scipy.stats import beta as beta_gen
from scipy import stats

def sigma_to_beta(mu, sd):
    # Calculate element wise values
    v1 = (mu * (1-mu))
    v2 = 1/sd**2
    # Multiply transpose to get each pairing of mean/sd calculated for v
    v = np.matmul(v1, v2.T) - 1
    # Calculate alpha and beta using element wise multiplication
    alp = mu*v
    bet = (1-mu)*v
    return alp, bet


class BMC:
    def __init__(self, max_batch_size=5000, alpha_resolution=1000, pdf_resolution=1000, resolution=200):
        self.max_batch_size = max_batch_size
        self.alpha_resolution = alpha_resolution # Number of values of alpha to evaluate over 
        self.pdf_resolution = pdf_resolution # Resolution of points along pdf to evaluate at 
        self.resolution = resolution

    def calculate_prior(self, utterances):
        total_mu = np.nanmean(utterances, axis=-1)
        total_sd = np.nanstd(utterances, axis=-1)

        kde_mu = stats.gaussian_kde(total_mu)
        x_mu = np.linspace(0, 1, self.resolution)
        p_mu = kde_mu(x_mu).reshape(self.resolution, 1)

        kde_sd = stats.gaussian_kde(total_sd)
        x_sd = np.linspace(0, 1, self.resolution)
        p_sd = kde_sd(x_sd).reshape(self.resolution, 1)

        prior = np.matmul(p_mu, p_sd.T)
        prior = prior / (self.resolution**2)

        self.prior = prior # Prior[ai, bi] = probability of mean=ai and std=bi

    def compute_inferred_distributions(self, alp, bet, ratings, x):
        ppdf = beta_gen.pdf(x[None,:], alp[:,None], bet[:,None])
        px = self.calculate_px(ratings, x, ppdf)

        post_prob_toy = np.nanprod(px, axis=-1)*self.active_prior[:,None]

        return self.all_post_calculation(post_prob_toy, ppdf), post_prob_toy

    def get_infer_distribution(self, utterances):
        x = np.linspace(0,1,self.pdf_resolution)
        x_mu = np.linspace(0,1,self.resolution).reshape(200,1)
        x_sd = np.linspace(0,1,self.resolution).reshape(200,1)

        # Calculate all possible alpha/beta parameters for beta distribution
        alp, bet = sigma_to_beta(x_mu, x_sd)

        # Remove possibilities where alpha and beta parameters are < 1 (not approximate normal distribution)
        alp[:,0] = 0
        bet[:,0] = 0
        alp[alp < 1] = 0
        bet[bet < 1] = 0

        # Now remove the invalid alpha and beta parameters
        # no point keeping full array in memory and removing invalid operations later
        msk = (alp>1) & (bet>1)
        self.active_prior = self.prior[msk]
        alp, bet = alp[msk], bet[msk]

        all_pdf, _ = self.compute_inferred_distributions(alp, bet, utterances, x)

        return all_pdf

    def find_arousal_indexs(self, all_pdfs, alp, bet, post_prob_toy, x):
        # Fast method
        ### Explanation ###
        # Calculate all possible intervals Ih
        # Now iterate through all possible P_alphas and find the associated Ih where the area/probability in Ih = P_alpha
        # i.e. for alpha = 1, the Ih will cover the full probability (it's kind of like a confidence interval)
        arousal_indexs, min_areas = [], []

        # Find the maximum point in the pdfs
        max_ids = np.nanargmax(all_pdfs, axis=-1)
        if (max_ids == 0).any():
            print(all_pdfs[:,max_ids])
            raise ValueError(f'Warning found {(max_ids == 0).sum()} pdf(s) with max point at index 0')
            print('BMC not well defined in this case as interval end point cannot be 0')
            print('Setting the max id to index 1 in these cases')
            max_ids[max_ids == 0] = 1
        bs = all_pdfs.shape[0] # Batch size

        # For all utterances we process as a batch the h values
        # i.e. for the value where max_id occurs at the lowest index, all other samples also need to calculate h values in these positions
        interval_h = np.full((bs, max_ids.max(), 2), np.nan)

        # During selection of minimum, we require that start point < max_id, and that end point > max_id
        # We use equality in comparison here as we want the mask to be true when invalid and false when valid
        past_maximum = np.arange(self.pdf_resolution) > max_ids.reshape(-1,1)
        before_maximum = np.arange(self.pdf_resolution) < max_ids.reshape(-1,1)

        past_maximum_values = np.full((bs, 1000-(max_ids.min()+1)), np.inf) # inf - anything is inf, so won't be selected at argmin 
        before_maximum_values = np.full((bs, max_ids.max()), -np.inf) # subtracting - inf becomes inf so won't be selected at argmin

        past_maximum_values[past_maximum[:,max_ids.min()+1:]] = all_pdfs[past_maximum]
        before_maximum_values[before_maximum[:,:max_ids.max()]] = all_pdfs[before_maximum]

        # Want to select all values where f(y) > h (where h = f(y)[h])
        # We know that h < max_id, so all values up to max_id will be greater than h
        # if we subtract h from all possible values, we want the minimum positive value
        # the absolute value will get larger once past the minimum positive value so we can argmin the absolute value
        # only used for argmin so to save on time and computation power use float16
        choices = abs(past_maximum_values[:,None,:] - before_maximum_values[:,:,None])
        # Choices dimensions now represent [num utterances, possible end points, possible start points]
        idxs = np.argmin(choices, axis=-1)

        # Concatenate into one array 
        # Get cdf for x for each valid alp/bet parameter (default values will be shape 6916 x 1000)
        all_areas = beta_gen.cdf(x[None,:], alp[:,None], bet[:,None])

        # ppt with default values will be shape 6916 x num utterances
        ppt = post_prob_toy
        ppt = ppt/ppt.sum(axis=0)
        ppt = ppt.transpose(1,0) # put batch dimension first
        post_area_new = np.matmul(ppt, all_areas)

        # total_areas = []
        # intervals = []
        arousal_indexs = []
        min_areas = []
        # Now loop through utterances and calculate areas and intervals
        for i, max_id in tqdm(enumerate(max_ids), desc='Finding alpha intervals', total=max_ids.shape[0]):
            lower_idxs = np.arange(max_id)
            lower_ids = x[lower_idxs]
            upper_idxs = idxs[i,:max_id]+max_ids.min()
            upper_ids = x[upper_idxs]
            A1 = post_area_new[i,lower_idxs]
            A2 = post_area_new[i,upper_idxs]
            interval_h = np.concatenate((lower_ids[:,None], upper_ids[:,None]), axis=1)
            # intervals.append(interval_h)
            total_area = A1 + (1 - A2)
            # total_areas.append(total_area)

            # Now find the intervals where area is as close as possible to P_alpha
            alpha = np.linspace(0,1,self.alpha_resolution)
            min_ids = np.argmin(abs(total_area.reshape(-1,1) - alpha), axis=0)
            arousal_index = interval_h[min_ids]
            min_area = total_area[min_ids]
            arousal_indexs.append(arousal_index)
            min_areas.append(min_area)

        return np.array(min_areas), np.array(arousal_indexs) # min_areas is p_alpha_predicted, arousal_indexs is alpha_likely_index

    def likely_area_inferred(self, all_pdfs, alp, bet, post_prob_toy, arousal_index):
        area_calculated_for = np.unique(arousal_index) # There are many duplicates in the arousal indexs, to save on compute time only calculate cdf for each item individually 
        post_area = beta_gen.cdf(area_calculated_for[None,:], alp[:,None], bet[:,None])
        ppt = post_prob_toy
        ppt = ppt/ppt.sum(axis=0)
        ppt = ppt.transpose(1,0) # put batch dimension first
        pan = np.matmul(ppt, post_area)

        # Now loop through utterances and calculate areas and intervals
        areas = np.zeros((arousal_index.shape[0], arousal_index.shape[1]))
        for i in tqdm(range(arousal_index.shape[0]), desc='Finding likely areas'):
            area_idxs = np.argmin(abs(area_calculated_for[None,None,:] - arousal_index[i,:,:,None]), axis=-1)        
            # When indexing a 2d array numpy requires the index shape is the same as the output shape 
            # so we need to broadcast the index by adding a arange column at the front for utterance
            final = pan[i, area_idxs]
            A1 = final[:,0]
            A2 = final[:,1]
            areas[i,:] = A1 + (1 - A2)
        # areas = A1 + (1 - A2)
        return areas

    def BMC_calculation_function_from_predictions(self, predictions, gt_ratings):
        x = np.linspace(0,1,1000)
        x_mu = np.linspace(0,1,self.resolution).reshape(200,1)
        x_sd = np.linspace(0,1,self.resolution).reshape(200,1)

        # Calculate all possible alpha/beta parameters for beta distribution
        alp, bet = sigma_to_beta(x_mu, x_sd)

        # Remove possibilities where alpha and beta parameters are < 1 (not approximate normal distribution)
        alp[:,0] = 0
        bet[:,0] = 0
        alp[alp < 1] = 0
        bet[bet < 1] = 0

        # Now remove the invalid alpha and beta parameters
        # no point keeping full array in memory and removing invalid operations later
        msk = (alp>1) & (bet>1)
        self.active_prior = self.prior[msk]
        alp, bet = alp[msk], bet[msk]

        # For very large values numpy will use too much ram so use a max chunk size 
        BMC = np.zeros((predictions.shape[0], self.alpha_resolution))
        num_chunks = int(np.ceil(predictions.shape[0]/self.max_batch_size))
        for chunk_idx in tqdm(range(num_chunks), desc='Calculating BMC over smaller batches'):
            chunk_start = chunk_idx*self.max_batch_size
            chunk_end = min(predictions.shape[0], (chunk_idx+1)*self.max_batch_size)
                
            y_all_pdfs, y_post_prob_toy = self.compute_inferred_distributions(alp, bet, predictions[chunk_start:chunk_end,:], x)
            y_hat_all_pdfs, y_hat_post_prob_toy = self.compute_inferred_distributions(alp, bet, gt_ratings[chunk_start:chunk_end,:], x)

            p_alpha_predicted, alpha_likely_index = self.find_arousal_indexs(y_all_pdfs, alp, bet, y_post_prob_toy, x)
            p_hat_alpha_inferred = self.likely_area_inferred(y_hat_all_pdfs, alp, bet, y_hat_post_prob_toy, alpha_likely_index)

            # Now return BMC calculation
            AA1 = 1-p_alpha_predicted
            AA2 = 1-p_hat_alpha_inferred
            chunk_BMC = AA2/AA1
            BMC[chunk_start:chunk_end] = chunk_BMC

        return BMC

    def calculate_px(self, rating, x, ppdf):
        # Return probability of the values in x 
        # Find the closest point in the calculated pdf to the real utterance ratings
        temp = abs(rating[:,:,None] - x)
        idx = np.argmin(temp, axis=-1)#.astype(np.int64) # if batch = 60000 and num annotators = 10 then; argmin(60000 x 10 x 1000, axis=-1) -> [60000,10]
        # The final dimension of temp will be all NaN if the annotator wasn't present in that utterance
        # In these cases we also want to mask px to NaN so we need to calculate this 
        px_nan_mask = np.isnan(np.min(temp, axis=-1))
        px = ppdf[:,idx] # alpha/beta parameter combinations x 1000 indexed by (60000 x 10) -> alpha/beta parameter combinations x 60000 x 10 

        # px will be of shape resolution X resolution X num utterances X num annotators
        # annotator values will be NaN when annotator not present in current utterance 
        # so remove these annotators
        # Want to apply the mask over entirety of first dimension so broadcast using None
        px[:,px_nan_mask] = np.nan

        # rating will contain np.nan when annotators are missing we want to remove these values now
        # This needs to be expanded to not be per utterance but batch of utterance using NaN
        # Have to add a small epsilon as we should never have 0 probability but in some cases px will assign a zero probability
        # when we later take the product of this output the value becomes 0 and invalid 
        return px + 1e-8 # Get result for all 200x200 distributions calculated

    def all_post_calculation(self, post, ppdf):
        post = (post/np.nansum(post, axis=0))
        # Want to sum over the first dimension, and additionally multiply the [num utterances] and [num pdf evaluation points] (the last dimensions in post and ppdf respectively)
        # we can do this all in one step saving time and memory by reducing first two dimensions into one shared dimension and then doing a matrix multiplication to reduce these dimensions
        all_pdf = np.matmul(post.reshape(-1, post.shape[-1]).T, ppdf.reshape(-1, ppdf.shape[-1]))
        return all_pdf
