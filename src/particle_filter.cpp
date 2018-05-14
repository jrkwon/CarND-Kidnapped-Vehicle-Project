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

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	particles.clear();
	weights.clear();

	num_particles = 100;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	default_random_engine gen;
	for (int i = 0; i < num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(particle.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	default_random_engine gen;

	double yaw = yaw_rate * delta_t;
	for (int i = 0; i < num_particles; i++)
	{
		if (fabs(yaw_rate) < 0.0001)
		{
			particles[i].x += (velocity * delta_t * cos(particles[i].theta));
			particles[i].y += (velocity * delta_t * sin(particles[i].theta));
			//particles[i].theta += yaw;
		}
		else
		{
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw));
			particles[i].theta += yaw;
		}

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (uint i = 0; i < observations.size(); i++)
	{
		// init min_dist with the largest number possible.
		double min_dist = numeric_limits<double>::max();
		int min_dist_landmark_id = -1;

		// find a particle that has the min dist to a landmark
		for (uint j = 0; j < predicted.size(); j++)
		{
			double diff_dist = dist(observations[i].x, observations[i].y,
									predicted[j].x, predicted[j].y);
			if (diff_dist < min_dist)
			{
				min_dist = diff_dist;
				min_dist_landmark_id = predicted[j].id;
			}
		}

		observations[i].id = min_dist_landmark_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
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

	double sum_weights = 0.0;

	for (int i = 0; i < num_particles; i++)
	{

		// 1. transform observations from car to map coordinate
		vector<LandmarkObs> trans_observations;
		for (uint j = 0; j < observations.size(); j++)
		{

			LandmarkObs obs;

			obs.id = j;
			obs.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
			obs.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);

			trans_observations.push_back(obs);
		}

		// 2. get landmarks that are only in the sensor range
		vector<LandmarkObs> predicted;
		for (uint j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			if ((fabs(particles[i].x - map_landmarks.landmark_list[j].x_f) <= sensor_range) && (fabs(particles[i].y - map_landmarks.landmark_list[j].y_f) <= sensor_range))
			{

				predicted.push_back(
					LandmarkObs{
						map_landmarks.landmark_list[j].id_i,
						map_landmarks.landmark_list[j].x_f,
						map_landmarks.landmark_list[j].y_f});
				//std::cout << "predicted: " << map_landmarks.landmark_list[j].id_i << ", " << map_landmarks.landmark_list[j].x_f << ", " << map_landmarks.landmark_list[j].y_f << std::endl;
			}
		}

		// 3. data association
		dataAssociation(predicted, trans_observations);

		// 4. update weights based on the multivariate Gassian probability function
		particles[i].weight = 1.0; // reset the weight
		double normalizer = 0.5 / (M_PI * std_landmark[0] * std_landmark[1]);
		double pred_x, pred_y;
		for (uint j = 0; j < trans_observations.size(); j++)
		{
			for (uint k = 0; k < predicted.size(); k++)
			{
				if (trans_observations[j].id == predicted[k].id)
				{
					pred_x = predicted[k].x;
					pred_y = predicted[k].y;
					break;
				}
			}

			double gaussian_prob = exp(-(
				(pow((trans_observations[j].x - pred_x), 2) / (2.0 * pow(std_landmark[0], 2))) + (pow((trans_observations[j].y - pred_y), 2) / (2.0 * pow(std_landmark[1], 2)))));
			particles[i].weight *= normalizer * gaussian_prob;
		}
		sum_weights += particles[i].weight;
	}

	// 5. normalize all weights
	for (int i = 0; i < num_particles; i++)
	{
		particles[i].weight /= sum_weights;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> new_particles;
	int index;

	// Initializes discrete distribution function
	std::random_device rd;
	std::mt19937 gen(rd()); // Mersenne Twister 19937 generator
	std::discrete_distribution<int> weight_distribution(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++)
	{
		index = weight_distribution(gen);
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
										 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
