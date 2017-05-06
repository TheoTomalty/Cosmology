from __future__ import division
import json
import getopt
import sys
import numpy as np
from algorithm.DirectoryEmbedded import *
import matplotlib.pyplot as plt
from scipy import stats
from scipy import special
import matplotlib

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)


def gauss(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

class Analysis(DirectoryEmbedded):
    def __init__(self, directory):
        DirectoryEmbedded.__init__(self, directory)
        self.dists = []
        self.tensions = []
        self.noise_dists = []
        
        self.begin = 10
        self.end = 40
    
    def noise_mean(self):
        combined_array = self.noise_dists[0]
        
        for i in range(1, len(self.noise_dists)):
            combined_array += self.noise_dists[i]
        
        l = np.array(combined_array)
        return np.mean(l), np.var(l)/len(l)
    
    def string_mean(self, hist_num):
        assert hist_num < len(self.dists)
        
        l = np.array(self.dists[hist_num])
        return np.mean(l), np.var(l)/len(l)
    
    def noise_var(self):
        combined_array = self.noise_dists[0]
        
        for i in range(1, len(self.noise_dists)):
            combined_array += self.noise_dists[i]
        
        mean, var = self.noise_mean()
        l = (np.array(combined_array) - mean)**2
        l2 = (np.array(combined_array) - mean)**4
        
        return np.mean(l), (np.mean(l2) - var**2)/len(l)
    
    def string_var(self, hist_num):
        assert hist_num < len(self.dists)
        
        mean, var = self.string_mean(hist_num)
        l = (np.array(self.dists[hist_num]) - mean)**2
        l2 = (np.array(self.dists[hist_num]) - mean)**4
        
        return np.mean(l), (np.mean(l2) - var**2)/len(l)
    
    def t_test(self, hist_num, noise=None):
        var_noise = self.noise_var()
        if noise is None:
            noise = self.noise_mean()
        
        y, var = self.string_mean(hist_num)
        t = (y - noise[0])/np.sqrt(var + noise[1])
        
        z, zar = self.string_var(hist_num)
        f = (z - var_noise[0])/np.sqrt(zar + var_noise[1])
        return t, f
    
    def print_t(self, log_axis=False, bounds=None):
        noise = self.noise_mean()
        
        t = []
        terr = []
        for i in range(len(self.dists)):
            t_i, f_i = self.t_test(i, noise=noise)
            terr_i = 1
            t.append(t_i)
            terr.append(terr_i)
            
            print t_i, f_i
            
        terr_bot = [min(np.abs(t[i]) - 0.1, terr[i]) for i in range(len(t))]
        
        plt.yscale('log', nonposy='clip')
        if log_axis:
            plt.xscale('log')
        plt.ylim(ymin=0.1, ymax=100)
        if log_axis:
            plt.xlim(xmin=bounds[0], xmax=bounds[1])
        plt.errorbar(np.array(self.tensions), np.abs(np.array(t)), yerr=(np.array(terr_bot), np.array(terr)), fmt='o')
        plt.ylabel("t-Test Significance", fontsize=20)
        plt.xlabel("$G\\mu$ ($\\times 10^{-8}$)", fontsize=30)
        plt.tight_layout()
        plt.savefig("results_P.png")
    
    def show_distributions(self, num_bins):
        bin_size = (self.end - self.begin)/num_bins
        inbins = np.arange(self.begin, self.end, bin_size)
        
        n, bins, patches = plt.hist(np.array(self.noise_dists[0]), bins=inbins, normed=0, facecolor='green', alpha=0.75)
        n1, bins1, patches1 = plt.hist(np.array(self.dists[-1]), bins=inbins, normed=0, facecolor='blue', alpha=0.75)
        
        #popt, pcov = curve_fit(
        #    gauss,
        #    bins[:-1], n,
        #    p0=(0.15, 18, 2)
        #)
        
        
        #plt.plot(bins[:-1], gauss(bins[:-1], *popt))
        plt.legend([patches[0], patches1[0]], ['No strings', '$G\\mu = 10^{-7}$'])
        
        plt.ylabel("Count", fontsize=20)
        plt.xlabel("Algorithm Output", fontsize=20)
        plt.tight_layout()
        plt.savefig('hist.png')
    
    def noise_hist(self, num_bins, exclude=None):
        bin_size = (self.end - self.begin)/num_bins
        inbins = np.arange(self.begin, self.end + 0.5*bin_size, bin_size)
        
        noise_hist = None
        num_hists = 0
        for hist, count in zip(self.noise_dists, range(1000)):
            if exclude == count:
                continue
            n, bins = np.histogram(np.array(hist), bins=inbins)
            num_hists += 1
            if noise_hist is not None:
                noise_hist += n
            else:
                noise_hist = n
        
        return noise_hist/num_hists, np.sqrt(noise_hist)/num_hists
    
    def string_hist(self, hist_num, num_bins, noise=False):
        bin_size = (self.end - self.begin)/num_bins
        inbins = np.arange(self.begin, self.end + 0.5*bin_size, bin_size)
        
        assert (hist_num < len(self.dists) and not noise) or (hist_num < len(self.noise_dists) and noise), "Invalid histogram index"
        string_hist, _ = np.histogram(np.array(
            (self.dists[hist_num] if not noise else self.noise_dists[hist_num])
        ), bins=inbins)
        
        return string_hist, np.sqrt(string_hist)
    
    def chi_test(self, hist_num, num_bins, noise=False):
        s, serr = self.string_hist(hist_num, num_bins, noise=noise)
        r, rerr = self.noise_hist(num_bins, exclude=(None if not noise else hist_num))
        
        chisq = 0
        ndf = 0
        for s_i, serr_i, r_i, rerr_i in zip(s, serr, r, rerr):
            if s_i > 2 or r_i > 2:
                sr_i = (serr_i if serr_i else 1)
                rr_i = (rerr_i if rerr_i else 1)
                chisq += (s_i - r_i)**2/(sr_i**2 + rr_i**2)
                ndf += 1
        
        p_val = 1.0 - stats.chi2.cdf(chisq, ndf)
        sigma = np.sqrt(2)*special.erfcinv(p_val)
        
        print str(hist_num + 1) + ": ", chisq, ndf, sigma
    
    def print_test(self, num_bins):
        for hist_num in range(len(self.dists)):
            self.chi_test(hist_num, num_bins)
    
    def print_control(self, num_bins):
        for hist_num in range(len(self.noise_dists)):
            self.chi_test(hist_num, num_bins, noise=True)
    
    def add_line(self, line, noise):
        data = json.loads(line)
        
        theta = data['theta']
        scalar, labels = np.array(data['scalar']), np.array(data['labels'])
        
        for value, label in zip(scalar.reshape(-1), labels.reshape(-1)):
            if not noise:
                self.dists[-1].append(value)
            else:
                self.noise_dists[-1].append(value)
    
    def load_file(self, file_name, noise):
        with open(file_name, 'r') as file:
            for line in file:
                self.add_line(line[:-1], noise)
                
    def load_directory(self, directory=None, noise=False, tension=1.0):
        if not noise:
            self.dists.append([])
            self.tensions.append(tension)
            
        else:
            self.noise_dists.append([])
        if directory is None:
            dir = self.directory
        else:
            dir = directory
        files = get_files(dir, "final")
        for file_name in files:
            self.load_file(file_name, noise)

if __name__ == "__main__":
    directory = ""
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:",[])
    except getopt.GetoptError:
        print "Invalid inputs"
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            #Input directory to save the numbered image files, ex: "images1.txt"
            if not os.path.exists(arg):
                print "Image Directory not found."
                sys.exit()
            else:
                directory = arg
    
    a = Analysis(directory)
    #a.load_directory("tmptest_none", noise=True)
    #a.load_directory("tmptest_none2", noise=True)
    #a.load_directory("tmptest_none3", noise=True)
    #a.load_directory("tmptest_none4", noise=True)
    #a.load_directory("tmptest_none5", noise=True)
    #a.load_directory("tmptest_none6", noise=True)
    #a.load_directory("tmptest_none7", noise=True)
    #a.load_directory("tmptest2_01", tension=1)
    #a.load_directory("tmptest2_02", tension=2)
    #a.load_directory("tmptest2_03", tension=3)
    #a.load_directory("tmptest2_04", tension=4)
    #a.load_directory("tmptest2_05", tension=5)
    #a.load_directory("tmptest2_06", tension=6)
    #a.load_directory("tmptest2_07", tension=7)
    #a.load_directory("tmptest2_08", tension=8)
    #a.load_directory("tmptest2_09", tension=9)
    #a.load_directory("tmptest2_10", tension=10)
    
    a.load_directory("poltest_none", noise=True)
    a.load_directory("poltest_none2", noise=True)
    a.load_directory("poltest_none3", noise=True)
    a.load_directory("poltest_none4", noise=True)
    a.load_directory("poltest_none5", noise=True)
    a.load_directory("poltest_none6", noise=True)
    a.load_directory("poltest_none7", noise=True)
    
    a.load_directory("poltest_06", tension=6)
    #a.load_directory("poltest_08", tension=8)
    a.load_directory("poltest_10", tension=10)
    a.load_directory("poltest_15", tension=15)
    a.load_directory("poltest_20", tension=20)
    a.load_directory("poltest_30", tension=30)
    a.load_directory("poltest_50", tension=50)
    
    #num_bins = 500
    #a.print_control(num_bins)
    #print " "
    #a.print_test(num_bins)
    #print " "
    a.print_t(log_axis=True, bounds=(4, 100))
    
    #a.show_distributions(100)
    
    #x = np.arange(1, 11)
    #t, terr = [], []
    #
    #mu_0 = 0
    #var = 0
    #for dist, index in zip(a.dists, range(20)):
    #    if index < 7:
    #        y = np.array(dist)
    #        mean = np.mean(y)
    #        mu_0 += mean/7
    #        var += np.mean((y-mean)**2)/(7*1600*25)/7
    #    else:
    #        y = np.array(dist)
    #        mean = np.mean(y)
    #        rms = np.sqrt(np.mean((y-mean)**2)/(1600*25))
    #        t.append((mean - mu_0)/np.sqrt(rms**2 + var))
    #        terr.append(rms/np.sqrt(rms**2 + var))
    #        print (mean - mu_0)/np.sqrt(rms**2 + var), rms/np.sqrt(rms**2 + var)
    #
    #plt.yscale('log', nonposy='clip')
    #plt.ylim(ymin=0.1, ymax=100)
    #terr[0] = np.abs(t[0]) - 0.1
    #plt.errorbar(x, np.abs(np.array(t)), yerr=np.array(terr), fmt='o')
    #plt.ylabel("t-Test significance")
    #plt.xlabel("String tension ($\\times 10^{-8}$)")
    #plt.savefig("results.png")
