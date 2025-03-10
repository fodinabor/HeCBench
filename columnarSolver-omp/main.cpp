/*
Copyright 1998–2018 Bernhard Esslinger and the CrypTool Team. Permission
is granted to copy, distribute and/or modify this document under the terms of the
GNU Free Documentation License, Version 1.3 or any later version published by the
Free Software Foundation (FSF). A copy of the license is included in the section
entitled "GNU Free Documentation License".
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define B ((int)32)
#define T ((int)32)
#define THREADS ((int)B*T)
#define CLIMBINGS 150000
#define ALPHABET 26
#define totalBigrams ((int)ALPHABET*ALPHABET)
#define CAP ((float)999999.0)

#define ENCRYPTED_T "tteohtedanisroudesereguwocubsoitoabbofeiaiutsdheeisatsarsturesuaastniersrotnesctrctxdiwmhcusyenorndasmhaipnnptmaeecspegdeislwoheoiymreeotbsspiatoanihrelhwctftrhpuunhoianunreetrioettatlsnehtbaecpvgtltcirottonesnobeeeireaymrtohaawnwtesssvassirsrhabapnsynntitsittchitoosbtelmlaouitrehhwfeiaandeitciegfreoridhdcsheucrnoihdeoswobaceeaorgndlstigeearsotoetduedininttpedststntefoeaheoesuetvmmiorftuuhsurof"
#define ENCRYPTEDLEN ((int)sizeof(ENCRYPTED_T)-1)

#define DECRYPTED_T "thedistinctionbetweentherouteciphertranspositionandthesubstitutioncipherwherewholewordsaresubstitutedforlettersoftheoriginaltextmustbemadeonthebasisofthewordsactuallyuseditisbettertoconsidersuchamessageasaroutecipherwhenthewordsusedappeartohavesomeconsecutivemeaningbearingonthesituationathandasubstitutioncipherofthisvarietywouldonlybeusedfortransmissionofashortmessageofgreatimportanceandsecrecy"

#define KEY_LENGTH 30
#define SECTION_CONSTANT ENCRYPTEDLEN/KEY_LENGTH

#define HEUR_THRESHOLD_OP1 50
#define HEUR_THRESHOLD_OP2 70

#define OP1_HOP 4
#define OP2_HOP 2


#include "kernels.cpp"

void extractBigrams(float *scores, const char* filename) {
  FILE* bigramsFile = fopen(filename, "r");
  while(1){
    char tempBigram[2];
    float tempBigramScore = 0.0;
    if (fscanf(bigramsFile, "%s %f", tempBigram, &tempBigramScore) < 2)
    { break; } 
    scores[(tempBigram[0]-'a')*ALPHABET + tempBigram[1]-'a'] = tempBigramScore; 
  }
  fclose(bigramsFile);
}

bool verify(int* encrMap) {
  bool pass = true;
  const char *expect = DECRYPTED_T;
  for (int j=0; j<ENCRYPTEDLEN; ++j) {
    if (encrMap[j] + 'a' != expect[j]) {
       pass = false; break;
    }
  }
  return pass;
}

float candidateScore(int* decrMsg, float* scores) {
  float total = 0.0;
  for (int j=0; j<ENCRYPTEDLEN-1; ++j) 
    total += scores[ALPHABET*decrMsg[j] + decrMsg[j+1]];  
  return total;
}


int main(int argc, char* argv[]) {

  const char* filename = argv[1];

  int encryptedMap[ENCRYPTEDLEN];

  for (int j=0; j<ENCRYPTEDLEN; ++j)
    encryptedMap[j] = ENCRYPTED_T[j] - 'a';

  float scores[totalBigrams];  
  extractBigrams(scores, filename);

  int* decrypted = new int[ENCRYPTEDLEN*THREADS];
  unsigned int state[THREADS];

#pragma omp target data map(to: scores[0:totalBigrams], \
                                encryptedMap[0:ENCRYPTEDLEN]) \
                        map(from: decrypted[0:ENCRYPTEDLEN * THREADS]) \
                        map(alloc: state[0:THREADS])
  {
    #pragma omp target teams distribute parallel for thread_limit(T)
    for (int idx = 0; idx < THREADS; idx++) {
      for (int i = 0; i < idx; i++)
        LCG_random_init(&state[idx]);
    }

    #pragma omp target teams num_teams(B) thread_limit(T)
    {
      float shared_scores[ALPHABET*ALPHABET];
      #pragma omp parallel 
      {
        decodeKernel(scores, encryptedMap, state, decrypted, shared_scores);
      }
    }
  }

  int bestCandidate = 0;
  float bestScore = CAP;
  float* scoreHistory = new float[B*T];

  //  Calculating the best score ..
  for (int j=0; j<THREADS; ++j)  {
    float currentScore = candidateScore(&decrypted[ENCRYPTEDLEN*j], scores);
    scoreHistory[j] = currentScore;
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = j;
    }
  }

  //printf("Best candidate score: %f\n", bestScore);
  bool pass = verify(&decrypted[ENCRYPTEDLEN*bestCandidate]);
  printf("%s\n", pass ? "PASS" : "FAIL");

  delete[] decrypted;
  delete[] scoreHistory;
  return 0;
}
