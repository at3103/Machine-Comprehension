import edu.stanford.nlp.trees.LabeledScoredTreeNode;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.LabeledScoredTreeNode;
import edu.stanford.nlp.trees.Constituent;
import edu.stanford.nlp.trees.PennTreeReader;
import edu.stanford.nlp.trees.TreeReader;
import edu.stanford.nlp.trees.LabeledScoredConstituentFactory;
import edu.stanford.nlp.trees.ConstituentFactory;
import java.util.List;
import java.util.Arrays;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.Iterator;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import java.lang.Exception;

public class  GetConstituentsQ{
	static String parsedDir = "/home/bhavana/Documents/machine-comprehension-ensemble-learning/sliding_window/parsed_";
	static String writeDir = "/home/bhavana/Documents/machine-comprehension-ensemble-learning/sliding_window/processed_";
	//static String parsedDir = "/home/bhavana/Documents/machine-comprehension-ensemble-learning/sliding_window/test";
	//static String writeDir = "/home/bhavana/Documents/machine-comprehension-ensemble-learning/sliding_window/processed_test";
	static final ConstituentFactory cFact = new LabeledScoredConstituentFactory();
	static JSONArray getSpans(JSONArray tokens){
		JSONArray spans = new JSONArray();
		for(int i=0;i<tokens.size();i++){
			for(int j=i;j<tokens.size();j++){
				JSONObject span = new JSONObject();
				span.put("start",i);
				span.put("end",j);
				StringBuffer cur_span = new StringBuffer();
				JSONArray cur_span_tokens = new JSONArray();
				for(int k=i;k<=j;k++){
					cur_span.append(tokens.get(k));
					cur_span.append(' ');
				}				
				span.put("text",cur_span.toString());
				spans.add(span);
			}
		}
		return spans;
	}
	static void parseConstituents(String fileName){
		try {     
			JSONParser parser = new JSONParser();
		
			String filePath = parsedDir + "/" + fileName;
			Object obj = parser.parse(new FileReader(filePath));

            JSONObject jsonObject =  (JSONObject) obj;

            JSONArray questions = (JSONArray) jsonObject.get("questions"); 
            JSONArray processedQuestions = new JSONArray(); 
            Iterator<JSONObject> qiterator = questions.iterator();
	            
            while(qiterator.hasNext()) {
	            JSONObject question = qiterator.next();

	            JSONArray sentences = (JSONArray) question.get("sentences");
	         	
	         	JSONObject processedObj = new JSONObject();

	         	Iterator<JSONObject> iterator = sentences.iterator();
	            JSONArray processedSentences = new JSONArray();

	            while (iterator.hasNext()) {
	            	JSONObject processedSentence = new JSONObject();
	            	JSONObject sentence =  iterator.next();
	            	String treeString = (String) sentence.get("parse");
	            	TreeReader r = new PennTreeReader(new StringReader(treeString));
	            	JSONArray tokens = (JSONArray) sentence.get("tokens");
	            	processedSentence.put("parse",sentence.get("parse"));
					processedSentence.put("tokens",sentence.get("tokens"));
					processedSentence.put("pos",sentence.get("pos"));
					processedSentence.put("deps_basic",sentence.get("deps_basic"));
					processedSentence.put("deps_cc",sentence.get("deps_cc"));
					processedSentence.put("lemmas",sentence.get("lemmas"));
					processedSentence.put("normner",sentence.get("normner"));
					processedSentence.put("ner",sentence.get("ner"));
					

					JSONArray processedConstituents = new JSONArray();

	             	Tree tree = r.readTree();
					for(Constituent str_consti : tree.constituents(cFact)) {
						JSONArray cur_constituent_tokens = new JSONArray();
						JSONObject constituentObj = new JSONObject();
						StringBuffer cur_constituent = new StringBuffer();
			          	for(int i=str_consti.start();i<=str_consti.end();i++) {
			            	cur_constituent.append(tokens.get(i).toString());
			            	cur_constituent.append(" ");
			            	cur_constituent_tokens.add(tokens.get(i).toString());	
			          	}
			          	constituentObj.put("start",str_consti.start());
			          	constituentObj.put("end",str_consti.end());
			          	constituentObj.put("text",cur_constituent.toString());	
			          	constituentObj.put("text_tokens",cur_constituent_tokens);	
			          	constituentObj.put("label",str_consti.label().toString()); 
			          	//System.out.println(str_consti.label());         	
			          	processedConstituents.add(constituentObj);
					}
					processedSentence.put("constituents",processedConstituents);
					//JSONArray spans = getSpans(tokens);
					//processedSentence.put("spans",spans);
		       		processedSentences.add(processedSentence);
	            	
	            }
	            processedObj.put("sentences",processedSentences);
	            processedObj.put("id",question.get("id"));
	            processedObj.put("answers",question.get("answers"));
	            
	            processedQuestions.add(processedObj);

	        }
	        
	        JSONObject allquestions = new JSONObject();
	        allquestions.put("questions",processedQuestions);
            writeIntoFile(allquestions,fileName);
            
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        } catch (Exception e){
        	e.printStackTrace();
        }

	}

	public static void writeIntoFile(JSONObject obj, String fileName){
		try {
			FileWriter file = new FileWriter(writeDir+"/"+fileName);
			file.write(obj.toJSONString());
			file.flush();
			file.close();
		} catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

	}
	public static void main(String[] args){
		if(args.length < 1){
			System.out.println("Provide dataset name to be processed");
			return;
		} else {
			System.out.println("Processed : "+args[0].toUpperCase());
			
			parsedDir = parsedDir + args[0]+"_q";
			writeDir = writeDir + args[0]+"_q";
		}
		File folder = new File(parsedDir);
		File[] listOfFiles = folder.listFiles();
		
		//listOfFiles.length
	    for (int i = 0; i < listOfFiles.length; i++) {
	      if (listOfFiles[i].isFile()) {
	        parseConstituents(listOfFiles[i].getName());
	   		} 

	   	  if((i+1)%100 == 0 ){
	   	  	System.out.println("Proccesed "+i+" files");
	   	  }
	    }
		
	}
}