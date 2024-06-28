

# %%

remote_config = {
    "DBGroupGPU": {
        "ip_address": "",
        "port": 0
    },
    # THU_spark10的配置
    "THU_spark10": {
        "ip_address": "172.6.31.12",
        "port": 30005
    },
    # 本地card_server的配置
    # "THU_spark08": {
    #     "ip_address": "localhost",
    #     "port": 30005
    # },
    # THU_spark08的配置
    "THU_spark08": {
        "ip_address": "166.111.121.55",
        "port": 20012
    }
}

# %%
from utility import utils
from utility.common_config import http_timeout, cal_timeout
import collections, time
import requests
import traceback

class RemoteConnector(object):
    def __init__(self, ip_address, port):
        '''
        
        '''
        self.ip_address = ip_address
        self.port = port
        self.start_tmpl = "http://{}:{}/start"
        self.stop_tmpl = "http://{}:{}/stop"
        self.ping_tmpl = "http://{}:{}/ping"
        self.cardinality_tmpl = "http://{}:{}/cardinality"
        self.cardinality_batch_tmpl = "http://{}:{}/cardinality_batch"

    def start(self, method, workload):
        """
        启动基数估计器
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        url = self.start_tmpl.format(self.ip_address, self.port)
        r = requests.get(url = url, 
            params = {"method": method,
                      "workload": workload}, timeout=20)

        return r.json()

    def stop(self, method):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        url = self.stop_tmpl.format(self.ip_address, self.port)
        r = requests.get(url = url,
            params = {"method": method}, timeout=http_timeout)

        return r.json()

    def ping(self,):
        """
        {Description}
        
        Args:
            None
        Returns:
            flag:
        """
        url = self.ping_tmpl.format(self.ip_address, self.port)
        r = requests.get(url = url, timeout=http_timeout)
        return r.json()


    def cardinality(self, method, sql_text, label):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        url = self.cardinality_tmpl.format(self.ip_address, self.port)
        r = requests.post(url=url,
            params = {"method": method},
            data = {
                "sql_text": sql_text,
                "label": label
            }, timeout=http_timeout)

        return r.json()

    @utils.timing_decorator
    def cardinality_batch(self, method, sql_list:list, label_list:list):
        """
        {Description}
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        # 
        num_threshold = 50000
        if len(sql_list) > num_threshold:
            print(f"cardinality_batch: Unexpected query number. len(sql_list) = {len(sql_list)}.")
            current_path = traceback.format_stack()
            
            # 打印当前路径
            print("Current function call path:")
            for path in current_path:
                print(path.strip())
            raise ValueError(f"cardinality_batch: Unexpected query number. len(sql_list) = {len(sql_list)}.")
        
        url = self.cardinality_batch_tmpl.\
            format(self.ip_address, self.port)

        # 尝试多次访问，并且在最后一次访问失败时打印当前的时间
        max_try_times, flag = 5, False
        
        for _ in range(max_try_times):
            try:
                r = requests.post(url=url,
                    params = {"method": method},
                    json = {
                        "sql_list": sql_list,
                        "label_list": label_list
                    }, timeout=cal_timeout(len(sql_list)))
                flag = True
                break
            except requests.exceptions.RequestException as e:
                # 请求失败，尝试再次发送
                time.sleep(3)
                continue

        if flag == False:
            # print("RemoteConnector.cardinality_batch: ")
            curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            err_str = f"RemoteConnector.cardinality_batch: network error. url = {url}. "\
                f"len(sql_list) = {len(sql_list)}. curr_time = {curr_time}."
            raise ValueError(err_str)
        
        # print(f"cardinality_batch: query_number = {len(sql_list)}. delta_time = {1000*(te - ts):.2f}(ms).")
        
        try:
            card_list = r.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"cardinality_batch: meet JSON error. sql_list = {sql_list}")
            raise e
        
        return card_list

class RemoteBase(object):
    def __init__(self, ip_address, port):
        '''
        构造函数
        '''
        self.rc = RemoteConnector(ip_address, port)
        self.method = "Test"
        # self.rc.start(method = self.method)

    def __del__(self,):
        '''
        析构函数
        '''
        # self.rc.stop(method = self.method)

    def launch_service(self, workload):
        """
        启动服务
        
        Args:
            arg1:
            arg2:
        Returns:
            res1:
            res2:
        """
        return self.rc.start(method = self.method, workload=workload)
        
    def test_connectivity(self,):
        """
        {Description}
    
        Args:
            arg1:
            arg2:
        Returns:
            return1:
            return2:
        """
        res = self.rc.ping()
        return res


    def get_cardinalities(self, sql_list:list, label_list:list):
        if len(sql_list) == 0:
            return []
        res = self.rc.cardinality_batch(self.method, sql_list, label_list)
        return res['card_list']

    def get_cardinality(self, sql_text:str, label:int):
        res = self.rc.cardinality(self.method, sql_text, label)
        return res['card']

# %%

class NeuroCardRemote(RemoteBase):
    '''
    NeuroCard方法的调用器
    '''
    def __init__(self, *args, **kwargs):
        '''
        构造函数
        '''
        super().__init__(*args, **kwargs)
        self.method = "NeuroCard"


class MSCNRemote(RemoteBase):
    '''
    NeuroCard方法的调用器
    '''
    def __init__(self, *args, **kwargs):
        '''
        构造函数
        '''
        super().__init__(*args, **kwargs)
        self.method = "MSCN"


class FCNRemote(RemoteBase):
    '''
    NeuroCard方法的调用器
    '''
    def __init__(self, *args, **kwargs):
        '''
        构造函数
        '''
        super().__init__(*args, **kwargs)
        self.method = "FCN"

class FCNPoolRemote(RemoteBase):
    '''
    NeuroCard方法的调用器
    '''
    def __init__(self, *args, **kwargs):
        '''
        构造函数
        '''
        super().__init__(*args, **kwargs)
        self.method = "FCNPool"

class FactorJoinRemote(RemoteBase):
    '''
    FactorJoin方法的调用器
    '''
    def __init__(self, *args, **kwargs):
        '''
        构造函数
        '''
        super().__init__(*args, **kwargs)
        self.method = "FactorJoin"

# %%
class DeepDBAdvanceRemote(RemoteBase):
    '''
    NeuroCard方法的调用器
    '''
    def __init__(self, option, *args, **kwargs):
        '''
        构造函数
        '''
        super().__init__(*args, **kwargs)
        if option == "jct":
            self.method = "DeepDB_jct"
        elif option == "rdc":
            self.method = "DeepDB_rdc"
        elif option == "naru":
            self.method = "DeepDB_naru"


    
# %%
