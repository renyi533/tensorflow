import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import com.alibaba.fastjson.JSON;
import graph_data_parser.Block;
import graph_data_parser.Meta;
import graph_data_parser.BlockParser;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FSDataInputStream;
import java.util.HashMap;
import java.nio.charset.StandardCharsets;

public class BinaryDataGen {
  public static int DEFAULT_PARTITION_CNT = 10;

  public static class EulerFormat extends FileOutputFormat<IntWritable, BytesWritable> {
  
    @Override
    public RecordWriter<IntWritable, BytesWritable> getRecordWriter(TaskAttemptContext job)
          throws IOException, InterruptedException {
      
      FileSystem fs = FileSystem.newInstance(job.getConfiguration());
      HashMap<Integer, FSDataOutputStream> out_stream_map = new HashMap<Integer, FSDataOutputStream>();
      String out_path = getOutputPath(job).toString();
  
      RecordWriter<IntWritable, BytesWritable> recordWriter = new RecordWriter<IntWritable, BytesWritable>() {
  
        @Override
        public void write(IntWritable key, BytesWritable value) throws IOException,
              InterruptedException {
              
          if (out_stream_map.get(key.get()) == null) {
            out_stream_map.put(key.get(), fs.create(new Path(out_path+"/data_"+key.get()+".dat"), true));
          } 
          FSDataOutputStream out_s = out_stream_map.get(key.get());
          byte[] bytes =  value.copyBytes();
          out_s.write(bytes);
        }
        
        @Override
        public void close(TaskAttemptContext context) throws IOException,
              InterruptedException {
          for (FSDataOutputStream s : out_stream_map.values()) {
            s.close();
          }
          fs.close();
        }
      };
      
      return recordWriter;
    }
  }

  public static class EulerMapper
       extends Mapper<Object, Text, IntWritable, BytesWritable>{

    private IntWritable n_key = new IntWritable(0);
    private Meta meta = null;
    private FSDataOutputStream out_s = null;
    private int partition_cnt = DEFAULT_PARTITION_CNT;
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      Block b = JSON.parseObject(value.toString(), Block.class);
      byte[] bytes = new BlockParser(meta).BlockJsonToBytes(b);
      n_key.set((int)Long.remainderUnsigned(b.getNode_id(), partition_cnt));
      BytesWritable n_value = new BytesWritable();
      n_value.set(bytes, 0, bytes.length);
      context.write(n_key, n_value); 
    }
    protected void  setup(org.apache.hadoop.mapreduce.Mapper.Context context
                          ) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      partition_cnt = conf.getInt("partition_cnt", DEFAULT_PARTITION_CNT);
      String in_path = FileInputFormat.getInputPaths(context)[0].toString();
      String meta_path = in_path + "/../" + "meta.json";
      FileSystem fs = FileSystem.newInstance(context.getConfiguration());
      FSDataInputStream s = fs.open(new Path(meta_path));
      String meta_info = s.readLine();
      meta = JSON.parseObject(meta_info, Meta.class);
      s.close();       
      fs.close();
    }
    protected void  cleanup(org.apache.hadoop.mapreduce.Mapper.Context context
                            ) throws IOException, InterruptedException {
    }
  }

  public static class EulerReducer
       extends Reducer<IntWritable,BytesWritable,IntWritable,BytesWritable> {
    protected void  setup(Context context
                          ) throws IOException, InterruptedException {
    }

    protected void  cleanup(Context context
                            ) throws IOException, InterruptedException {
    }

    public void reduce(IntWritable key, Iterable<BytesWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      for (BytesWritable val: values) {   
        context.write(key, val);   
      }
    }
  }

  public static class MyTool extends Configured implements Tool {  
  
    public final int run(final String[] args) throws Exception { 
      Configuration conf = super.getConf();
      String extraArgs[] = new GenericOptionsParser(conf, args).getRemainingArgs();
      int partition_cnt = DEFAULT_PARTITION_CNT;
      if (extraArgs.length >= 3) {
        partition_cnt = Integer.parseInt(extraArgs[2]);
      }
      conf.setInt("partition_cnt", partition_cnt);

      Job job = Job.getInstance(conf);
      job.setJarByClass(BinaryDataGen.class);
      job.setMapperClass(EulerMapper.class);
      job.setCombinerClass(EulerReducer.class);
      job.setReducerClass(EulerReducer.class);
      job.setOutputKeyClass(IntWritable.class);
      job.setOutputValueClass(BytesWritable.class);
      if (extraArgs.length < 2) {
        System.out.println("Please specify input/output directories");
        return 0;
      }
      FileInputFormat.addInputPath(job, new Path(extraArgs[0]));
      FileOutputFormat.setOutputPath(job, new Path(extraArgs[1]));
      job.setJobName("Json to Binary Data Conversion for Euler");

      LazyOutputFormat.setOutputFormatClass(job, EulerFormat.class);
      return (job.waitForCompletion(true) ? 0 : 1);
    }
  }  

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    int res = ToolRunner.run(new MyTool(), args);  
    System.exit(res);  
  }
}
