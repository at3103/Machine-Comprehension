import java.util.Properties;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.CorefChain.CorefMention;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;

/** A simple corenlp example ripped directly from the Stanford CoreNLP website using text from wikinews. */
public class Coref {

   public static void main(String[] args) throws Exception {
    Annotation document = new Annotation("Barach Obama was born in 1992. He was the president. His wife is Michele Obama. She is beautiful. He lives in the white house. It is in Washington.");
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,dcoref");
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    pipeline.annotate(document);
    System.out.println("---");
    System.out.println("coref chains");
    List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

    Map<Integer, CorefChain> corefs = document.get(CorefCoreAnnotations.CorefChainAnnotation.class);
    
    List<String> resolved = new ArrayList<String>();


    for (CorefChain cc : corefs.values()) {
      System.out.println("\t" + cc);
    }
    for (CoreMap sentence : sentences) {
      List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
      for (CoreLabel token : tokens) {
            Integer corefClustId= token.get(CorefCoreAnnotations.CorefClusterIdAnnotation.class);
            

            CorefChain chain = corefs.get(corefClustId);
            if(chain==null){
                resolved.add(token.word());
            } else{
                System.out.println(token.word() +  " --> corefClusterID = " + corefClustId);
                System.out.println("matched chain = " + chain);
            
                int sentIndx = chain.getRepresentativeMention().sentNum -1;
                CoreMap corefSentence = sentences.get(sentIndx);
                List<CoreLabel> corefSentenceTokens = corefSentence.get(TokensAnnotation.class);
                String newwords = "";
                CorefMention reprMent = chain.getRepresentativeMention();
                System.out.println("Replacing "+ token + " with"+reprMent);
                if (token.sentIndex() != sentIndx || token.index() < reprMent.startIndex || token.index() > reprMent.endIndex) {
                  for (int i = reprMent.startIndex; i < reprMent.endIndex; i++) {
                      CoreLabel matchedLabel = corefSentenceTokens.get(i - 1); 
                      resolved.add(matchedLabel.word());

                      newwords += matchedLabel.word() + " ";

                  }
                } else {
                  resolved.add(token.word());
                }
            }
 
      }
    }

    String resolvedStr ="";
    System.out.println();
    for (String str : resolved) {
        resolvedStr+=str+" ";
    }
    System.out.println(resolvedStr);
  }

}