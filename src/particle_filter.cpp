/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;
	vechile_direction = theta;

	default_random_engine rand_value;

  	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Init particles and weights
	for(int i=0; i<num_particles; ++i){
      	Particle p;
      	p.id = i;
      	p.x = dist_x(rand_value);
		p.y = dist_y(rand_value);
		p.theta = dist_theta(rand_value);
      	p.weight = 1;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine rand_value;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

    for(int i=0; i<num_particles; ++i){
      	double v_div_t = velocity/yaw_rate;
		double previouse_theta = particles[i].theta;

      	particles[i].theta = particles[i].theta + yaw_rate * delta_t + dist_theta(rand_value);
		particles[i].x = particles[i].x + v_div_t * (sin(particles[i].theta) - sin(previouse_theta)) + dist_x(rand_value);
		particles[i].y = particles[i].y + v_div_t * (cos(previouse_theta) - cos(particles[i].theta)) + dist_y(rand_value);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double max_weight = 0;

	// Transrate OBS to each observation
    for(int i; i<num_particles; ++i){
      	vector<double> translated_obs_x; // vector of translated obs x element
		vector<double> translated_obs_y; // vector of translated obs y element
        vector<int> associations;
      	weights.clear();

       	// Transrate single OBS and get nearest landmark
		for(int obs_index; obs_index<observations.size(); ++obs_index){
            double heading = vechile_direction - particles[i].theta;
			double x = particles[i].x + observations[i].x * cos(heading) - particles[i].y * sin(heading);
			double y = particles[i].y + observations[i].x * sin(heading) + particles[i].y * cos(heading);
            translated_obs_x.push_back(x);
			translated_obs_y.push_back(y);

            //associations.push_back(GetNearestLandmark(x, y, map));
          	Map::single_landmark_s nearest_landmark = GetNearestLandmark(x, y, map_landmarks);
			associations.push_back(nearest_landmark.id_i);

			double weight = multivariate_gausian(x, y, nearest_landmark.x_f, nearest_landmark.y_f);
          	weights.push_back(weight);
		}
		SetAssociations(particles[i], associations, translated_obs_x, translated_obs_y);

		// Calculate the particle final weight
		double final_weight = 1;
		for(int i=0; i<weights.size(); ++i){
			final_weight *= weights[i];
		}

		particles[i].weight = final_weight;

		if(final_weight > max_weight){
			vechile_direction = particles[i].theta;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> resampled_particles;
	default_random_engine rand_value;

	weights.clear();
	for(int i=0; i<num_particles; ++i) {
      	weights.push_back(particles[i].weight);
	}

	std::discrete_distribution<> dist(weights.begin(), weights.end());

  	for(int i=0; i<num_particles; ++i) {
		int index = dist(rand_value);
		resampled_particles.push_back(particles[index]);
	}

  	particles.clear();
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

Map::single_landmark_s ParticleFilter::GetNearestLandmark(double x, double y, Map map) {
	double min_distance = 1000000;
	//int nearest_landmark = 0;
	Map::single_landmark_s nearest_landmark;

	for(int i=0; i<map.landmark_list.size(); ++i){
		double distance = dist(x, y, map.landmark_list[i].x_f, map.landmark_list[i].y_f);
        if(distance < min_distance){
          	min_distance = distance;
			//nearest_landmark = map.landmark_list[i].id_i;
			nearest_landmark = map.landmark_list[i];
		}
	}

	return nearest_landmark;
}
